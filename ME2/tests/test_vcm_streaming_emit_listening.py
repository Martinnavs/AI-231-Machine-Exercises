"""`--emit-listening`: gate period events become JSONL `listening` records
on stdout (the schema `app.forward` reads), independent of `--log-periods`."""

from __future__ import annotations

import json

from me2_voicegen.common.features import SAMPLE_RATE
from me2_voicegen.vcm.streaming import __main__ as cli
from me2_voicegen.vcm.streaming.config import StreamingConfig
from me2_voicegen.vcm.streaming.gate import GateState
from me2_voicegen.vcm.streaming.policy import PolicyDecision


def _records(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_default_off_and_cli_flag():
    assert StreamingConfig().emit_listening is False
    args = cli.build_arg_parser().parse_args(["--emit-listening"])
    assert args.emit_listening is True


def test_sink_none_when_nothing_requested():
    assert cli._make_period_sink(False, False) is None


def test_sink_emits_active_then_passive(capsys):
    sink = cli._make_period_sink(False, True)
    sink("open", SAMPLE_RATE * 2, None)
    sink("closed", SAMPLE_RATE * 4, PolicyDecision(accept=False, reason="endpointed: timed out"))
    captured = capsys.readouterr()
    recs = [json.loads(line) for line in captured.out.splitlines()]
    assert [(r["event"], r["state"], r["t_seconds"]) for r in recs] == [
        ("listening", "active", 2.0),
        ("listening", "passive", 4.0),
    ]
    assert captured.err == ""  # digest stays off without --log-periods


def test_reopened_is_active_and_digest_still_called(capsys, monkeypatch):
    digest = []
    monkeypatch.setattr(cli, "_print_period_event", lambda *a: digest.append(a[0]))
    sink = cli._make_period_sink(True, True)
    sink("reopened", SAMPLE_RATE, None)
    assert json.loads(capsys.readouterr().out)["state"] == "active"
    assert digest == ["reopened"]


def test_run_end_with_open_period_emits_passive(capsys):
    class Inner:
        def poll(self, samples_seen, window=None):
            return GateState(is_open=True, open_at_samples=0)

        def close(self):
            pass

    gate = cli._RunEndGateLogger(Inner(), log=False, emit=True)
    gate.poll(SAMPLE_RATE * 3)
    gate.close()
    assert _records(capsys) == [{"event": "listening", "state": "passive", "t_seconds": 3.0}]


def test_forwarder_accepts_emitted_record():
    from app.forward import record_events

    line = json.dumps({"event": "listening", "state": "active", "t_seconds": 1.0})
    assert record_events(json.loads(line)) == [(1.0, "/api/listening", {"state": "active"})]
