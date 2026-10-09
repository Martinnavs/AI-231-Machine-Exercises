"""Model-side forwarder for the UI site (feature `ui-site`).

Reads the streaming runner's stdout JSONL (live, piped) or a replay file
and POSTs each model->UI event to the UI's HTTP API. It is a thin,
dependency-free client -- stdlib only, no `me2_voicegen` import. It is
the model side's *only* touch point with the UI, which keeps the
one-way model -> UI boundary clean: terminal 1 posts, terminal 2 serves.

Live (terminal 1, piped from the runner):

    python -m me2_voicegen.vcm.streaming ... | python -m app.forward

Replay (a recorded JSONL such as phase1_podcast5.jsonl):

    python -m app.forward --file phase1_podcast5.jsonl
    python -m app.forward --file phase1_podcast5.jsonl --realtime --synthesize-listening

`--realtime` paces the replay by each record's `t_seconds` (relative to
the first forwarded event); `--synthesize-listening` (file mode) adds
the listening-state events a listening gate would have emitted --
`active` at `t_seconds - 3` (the 3-second gate window) and `passive` at
`t_seconds` -- so the indicator demo works end-to-end in replay before
the model-side ticket (`.scratch/ui-site/tickets/
01-model-facing-gate-and-listening.md`) lands. In live mode nothing is
synthesized: until that ticket, the runner emits no listening events and
the indicator simply stays passive.

`window` records (and malformed lines) are ignored. A POST failure is
logged to stderr and the forwarder keeps going.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Iterable

LISTENING_LEAD_S = 3.0  # the gate's listening window; "active" starts this long before the trigger


def parse_line(line: str) -> dict | None:
    """One JSONL line -> record dict, or None for blank/malformed/non-dict."""
    line = line.strip()
    if not line:
        return None
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def record_events(
    record: dict, synthesize: bool = False
) -> list[tuple[float, str, dict]]:
    """The `(t_seconds, endpoint, payload)` POSTs one record produces.

    - `trigger`   -> one `/api/command` POST with the decoded intent/slots.
    - `listening` -> one `/api/listening` POST (model-side, ticket 01).
    - `window` and any other event type -> nothing.

    With `synthesize=True` (replay only), a `trigger` also produces the
    listening events a gate would have emitted around it.
    """
    event = record.get("event")
    if event == "trigger":
        t = float(record.get("t_seconds", 0.0))
        events = [
            (t, "/api/command", {"intent": record.get("intent"), "slots": record.get("slots") or {}})
        ]
        if synthesize:
            events.append((t - LISTENING_LEAD_S, "/api/listening", {"state": "active"}))
            events.append((t, "/api/listening", {"state": "passive"}))
        return events
    if event == "listening":
        t = float(record.get("t_seconds", 0.0))
        return [(t, "/api/listening", {"state": record.get("state")})]
    return []


def build_events(
    lines: Iterable[str], synthesize: bool = False
) -> list[tuple[float, str, dict]]:
    """Every line -> a time-sorted list of POSTs (blank/malfiltered lines
    and `window` records drop out)."""
    events: list[tuple[float, str, dict]] = []
    for line in lines:
        record = parse_line(line)
        if record is None:
            continue
        events.extend(record_events(record, synthesize=synthesize))
    events.sort(key=lambda item: item[0])
    return events


def post_event(base_url: str, endpoint: str, payload: dict, timeout: float = 5.0) -> bool:
    """POST one event; log-and-continue on failure. Returns success."""
    url = base_url.rstrip("/") + endpoint
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            ok = 200 <= response.status < 300
            status = response.status
    except (urllib.error.URLError, OSError) as exc:
        print(f"forward: POST {endpoint} {payload!r} failed: {exc}", file=sys.stderr)
        return False
    if not ok:
        print(f"forward: POST {endpoint} {payload!r} -> HTTP {status}", file=sys.stderr)
    return ok


def replay(
    events: list[tuple[float, str, dict]], base_url: str, realtime: bool
) -> int:
    """POST every event; pace by `t_seconds` when `realtime`. Returns the
    number of failed POSTs."""
    failures = 0
    if not events:
        return failures
    start = time.monotonic()
    t0 = events[0][0]
    for t, endpoint, payload in events:
        if realtime:
            delay = (t - t0) - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
        if not post_event(base_url, endpoint, payload):
            failures += 1
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward the streaming runner's JSONL events to the UI site.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="UI base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--file",
        help="replay a recorded JSONL file instead of reading stdin",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace replay by each record's t_seconds (relative to the first event)",
    )
    parser.add_argument(
        "--synthesize-listening",
        action="store_true",
        help="file mode: synthesize gate listening events around triggers "
        "(active at t-3s, passive at t) for the indicator demo",
    )
    args = parser.parse_args(argv)

    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            lines = handle.readlines()
        events = build_events(lines, synthesize=args.synthesize_listening)
        return 1 if replay(events, args.url, args.realtime) else 0

    # Live mode: forward each record as it arrives (natural real-time pace).
    for line in sys.stdin:
        record = parse_line(line)
        if record is None:
            continue
        for _t, endpoint, payload in record_events(record):
            post_event(args.url, endpoint, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
