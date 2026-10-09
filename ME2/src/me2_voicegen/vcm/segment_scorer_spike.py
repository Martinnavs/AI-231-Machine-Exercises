"""Reproducible diagnostic for one-inference, CTC-aligned command scoring.

Run with ``python -m me2_voicegen.vcm.segment_scorer_spike``.  This is a
scratch-audio evaluation harness, not a production streaming policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from me2_voicegen.common.features import SAMPLE_RATE, LogMelFeatureExtractor
from me2_voicegen.vcm.decoder import prefix_beam_search
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR
from me2_voicegen.vcm.pipeline import load_checkpoint, logp_for_waveform
from me2_voicegen.vcm.segment_scorer import force_align, segment_scores

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIO_DIR = PROJECT_ROOT / ".scratch" / "batch-period" / "scorer-spike-audio"
DEFAULT_OUT_DIR = PROJECT_ROOT / ".scratch" / "batch-period" / "scorer-spike-results"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "out" / "vcm" / "optionb-optionc" / "checkpoints" / "checkpoint.pt"
PERIOD_SECONDS = 3.0
NEGATIVE_STRIDE_SECONDS = 0.5
BEAM_WIDTH = 25

POSITIVES: dict[str, tuple[str, str, dict[str, str]]] = {
    "lights-on": ("lights on", "LIGHT_ON", {}),
    "pause": ("pause", "PAUSE", {}),
    "play-music": ("play music", "PLAY_MUSIC", {}),
    "remind-me-exercise": ("remind me to exercise", "CREATE_REMINDER", {"TASK": "exercise"}),
    "set-brightness-100-percent": ("set the brightness to 100 percent", "BRIGHTNESS", {"PERCENT": "100 percent"}),
    "set-temperature-22-degrees": ("set the temperature to 22 degrees", "TEMPERATURE", {"DEGREES": "22 degrees"}),
}
METRICS = ("decode_confidence", "full_path_mean", "segment_path_mean", "token_blank_margin")


@dataclass(frozen=True)
class ExampleResult:
    source: str
    kind: str
    placement: str
    expected_text: str | None
    expected_intent: str | None
    predicted_text: str | None
    predicted_intent: str | None
    correct_candidate: bool
    alignment_start_frame: int | None
    alignment_end_frame: int | None
    decode_confidence: float | None
    full_path_mean: float | None
    segment_path_mean: float | None
    token_blank_margin: float | None


def load_mono_16k(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0) if waveform.dim() == 2 else waveform
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    return waveform.contiguous()


def crop_outer_excess_by_energy(waveform: torch.Tensor, period_samples: int) -> torch.Tensor:
    """Fit overlong positive audio by dropping only outer samples.

    Recordings at or below a period are returned bit-for-bit.  For the few
    recordings that exceed it, evaluate all 10 ms-aligned contiguous period
    regions and retain the one with greatest waveform energy.  This is
    deterministic, never changes a source WAV, and cannot remove an internal
    pause because the retained region is contiguous.
    """
    if len(waveform) <= period_samples:
        return waveform
    hop = round(0.01 * SAMPLE_RATE)
    starts = list(range(0, len(waveform) - period_samples + 1, hop))
    last = len(waveform) - period_samples
    if starts[-1] != last:
        starts.append(last)
    start = max(starts, key=lambda offset: float(waveform[offset : offset + period_samples].square().sum()))
    return waveform[start : start + period_samples]


def place_in_period(waveform: torch.Tensor, placement: str, period_samples: int) -> torch.Tensor:
    if len(waveform) > period_samples:
        raise ValueError(f"trimmed waveform is {len(waveform) / SAMPLE_RATE:.3f}s, exceeds period")
    start = {"left": 0, "center": (period_samples - len(waveform)) // 2, "right": period_samples - len(waveform)}[placement]
    result = torch.zeros(period_samples, dtype=waveform.dtype)
    result[start : start + len(waveform)] = waveform
    return result


def negative_periods(waveform: torch.Tensor, period_samples: int, stride_samples: int) -> list[torch.Tensor]:
    if len(waveform) <= period_samples:
        return [place_in_period(waveform, "left", period_samples)]
    starts = list(range(0, len(waveform) - period_samples + 1, stride_samples))
    last = len(waveform) - period_samples
    if starts[-1] != last:
        starts.append(last)
    return [waveform[start : start + period_samples] for start in starts]


def decode_candidate(logp: np.ndarray) -> tuple[str | None, str | None, dict, float | None]:
    """The existing decoder's permissive terminal selection, once per period.

    ``DecodeResult`` intentionally does not expose its trie prefix. The spike
    needs that prefix for forced alignment, so it performs the same final-beam
    terminal selection locally rather than calling ``decode()`` and then
    rerunning an identical beam solely to recover the phrase.
    """
    best: tuple[str, str, dict, float] | None = None
    for prefix, entry in prefix_beam_search(logp, OPTIONB_GRAMMAR.root, beam_width=BEAM_WIDTH).items():
        if entry.node.terminal is None:
            continue
        score = entry.total() / len(logp)
        if best is None or score > best[3]:
            intent, slots = entry.node.terminal[0]
            best = (prefix, intent, slots, score)
    return best if best is not None else (None, None, {}, None)


def is_correct_candidate(intent: str | None, slots: dict, expected_intent: str | None, expected_slots: dict) -> bool:
    """Compare the grammar's semantic result, not a numeric surface form."""
    return intent == expected_intent and slots == expected_slots


def score_period(model, extractor, waveform: torch.Tensor, source: str, kind: str, placement: str, expected: tuple[str, str, dict[str, str]] | None) -> ExampleResult:
    logp = logp_for_waveform(model, extractor, waveform)
    predicted_text, predicted_intent, predicted_slots, decode_confidence = decode_candidate(logp)
    expected_text, expected_intent, expected_slots = expected if expected else (None, None, {})
    # Option B deliberately accepts both digit and spelled numeric surfaces;
    # evaluation therefore compares the grammar's semantic output rather than
    # penalizing a valid ``one hundred`` decode against a ``100`` filename.
    correct = is_correct_candidate(predicted_intent, predicted_slots, expected_intent, expected_slots)
    if predicted_text is None:
        return ExampleResult(source, kind, placement, expected_text, expected_intent, None, None, False, None, None, None, None, None, None)
    alignment = force_align(logp, predicted_text)
    scores = segment_scores(logp, alignment)
    return ExampleResult(source, kind, placement, expected_text, expected_intent, predicted_text, predicted_intent, correct, scores.start_frame, scores.end_frame, decode_confidence, scores.full_path_mean, scores.segment_path_mean, scores.token_blank_margin)


def threshold_sweep(results: list[ExampleResult], metric: str) -> list[dict]:
    scores = sorted({getattr(row, metric) for row in results if getattr(row, metric) is not None}, reverse=True)
    if not scores:
        return []
    sweep = []
    positives = [row for row in results if row.kind == "positive"]
    negatives = [row for row in results if row.kind == "negative"]
    for threshold in scores:
        accepted = lambda row: getattr(row, metric) is not None and getattr(row, metric) >= threshold
        true_accepts = sum(accepted(row) and row.correct_candidate for row in positives)
        false_accepts = sum(accepted(row) for row in negatives)
        sweep.append({"threshold": threshold, "positive_correct_accepts": true_accepts, "positive_recall": true_accepts / len(positives), "false_accepts": false_accepts, "negative_windows": len(negatives)})
    return sweep


def choose_threshold(sweep: list[dict]) -> dict | None:
    eligible = [row for row in sweep if row["false_accepts"] == 0]
    if not eligible:
        return None
    best_recall = max(row["positive_recall"] for row in eligible)
    return max((row for row in eligible if row["positive_recall"] == best_recall), key=lambda row: row["threshold"])


def render_markdown(results: list[ExampleResult], selections: dict[str, dict | None]) -> str:
    positives = [row for row in results if row.kind == "positive"]
    negatives = [row for row in results if row.kind == "negative"]
    lines = ["# Single-inference CTC segment-scorer spike", "", "This diagnostic makes one acoustic inference per 3.0-second period. It is not a production threshold calibration.", "", f"- Positive placement cases: {len(positives)}", f"- Negative windows: {len(negatives)}", "- Selection: zero false accepts first; then maximum correct positive recall; then stricter threshold.", "", "| metric | threshold | correct-positive recall | false accepts |", "| --- | ---: | ---: | ---: |"]
    for metric, selected in selections.items():
        if selected is None:
            lines.append(f"| `{metric}` | — | — | no zero-FA threshold |")
        else:
            lines.append(f"| `{metric}` | {selected['threshold']:.6f} | {selected['positive_correct_accepts']}/{len(positives)} ({selected['positive_recall']:.1%}) | 0/{len(negatives)} |")
    lines.extend(["", "## Per-positive cases", "", "| source | placement | expected | predicted | correct candidate | segment frames |", "| --- | --- | --- | --- | --- | --- |"])
    for row in positives:
        frames = "—" if row.alignment_start_frame is None else f"{row.alignment_start_frame}–{row.alignment_end_frame}"
        lines.append(f"| `{row.source}` | {row.placement} | {row.expected_text} | {row.predicted_text or 'REJECTED'} | {row.correct_candidate} | {frames} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    period_samples = round(PERIOD_SECONDS * SAMPLE_RATE)
    stride_samples = round(NEGATIVE_STRIDE_SECONDS * SAMPLE_RATE)
    model, _ = load_checkpoint(args.checkpoint, device="cpu", allow_unsafe_load=False)
    extractor = LogMelFeatureExtractor()
    results: list[ExampleResult] = []
    for stem, expected in POSITIVES.items():
        for path in sorted(args.audio_dir.glob(f"{stem}-*.wav")):
            trimmed = crop_outer_excess_by_energy(load_mono_16k(path), period_samples)
            for placement in ("left", "center", "right"):
                results.append(score_period(model, extractor, place_in_period(trimmed, placement, period_samples), path.name, "positive", placement, expected))
    expected_count = len(POSITIVES) * 3 * 3
    if len([row for row in results if row.kind == "positive"]) != expected_count:
        raise ValueError(f"expected {expected_count} positive placement cases; found a missing or extra positive WAV")
    for path in sorted(args.audio_dir.glob("room-tone.wav")) + sorted(args.audio_dir.glob("unrelated-voice-*.wav")):
        for i, period in enumerate(negative_periods(load_mono_16k(path), period_samples, stride_samples)):
            results.append(score_period(model, extractor, period, f"{path.name}#{i}", "negative", "window", None))
    sweeps = {metric: threshold_sweep(results, metric) for metric in METRICS}
    selections = {metric: choose_threshold(sweep) for metric, sweep in sweeps.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"period_seconds": PERIOD_SECONDS, "negative_stride_seconds": NEGATIVE_STRIDE_SECONDS, "results": [asdict(row) for row in results], "sweeps": sweeps, "selection": selections}
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out_dir / "results.md").write_text(render_markdown(results, selections))
    print(f"wrote {args.out_dir / 'results.json'} and {args.out_dir / 'results.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
