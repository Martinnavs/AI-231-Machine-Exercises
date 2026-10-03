"""Hybrid whole-clip decode: grammar-constrained CTC answer first, classifier-head answer as the fallback.

Policy `fallback` (A): the CTC decode (threshold, incomplete-prefix margin) answers when it accepts; otherwise the
intent head answers when its max-softmax clears `cls_threshold`. Policy `agree` (B): the intent head gates acceptance;
the CTC answer is used when it names the same intent, else the head's answer. The CTC and the heads may come from the
same checkpoint (one encoder pass) or from two (`cls_model`). Thresholds are arguments, never fitted here.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import torch

from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.decoder import decode
from me2_voicegen.vcm.semantic_labels import EXTRA_INTENT_CLASSES, INTENT_CLASSES, SLOTS, slot_head_name


@dataclasses.dataclass
class HybridDecision:
    intent: str | None
    slots: dict
    source: Literal["ctc", "cls", None]


def _combine(d, name: str, cls_ok: bool, cls_slots: dict, policy: str) -> HybridDecision:
    ctc_slots = {k: str(v) for k, v in d.slots.items()}
    ctc_ok = d.intent is not None
    if policy == "fallback":
        if ctc_ok:
            return HybridDecision(d.intent, ctc_slots, "ctc")
        return HybridDecision(name, cls_slots, "cls") if cls_ok else HybridDecision(None, {}, None)
    if policy != "agree":
        raise ValueError(f"unknown policy {policy!r}")
    if not cls_ok:
        return HybridDecision(None, {}, None)
    if ctc_ok and d.intent == name:
        return HybridDecision(d.intent, ctc_slots, "ctc")
    return HybridDecision(name, cls_slots, "cls")


@torch.no_grad()
def _forward(ctc_model, cls_model, feature_extractor, waveform, grammar, *, ctc_threshold, margin, cls_threshold, beam_width):
    """Encoder pass(es), one beam search, and the head readout: `(DecodeResult, intent name, confidence, accepted, slots)`."""
    features = feature_extractor(waveform).unsqueeze(0)
    if cls_model is ctc_model:
        head = ctc_model.forward_heads(features)
        logits = head.ctc_logits
    else:
        logits, head = ctc_model(features), cls_model.forward_heads(features)
    d = decode(logits.log_softmax(-1)[0].numpy(), grammar, ctc_threshold, beam_width=beam_width, required_command_margin=margin)
    probs = head.intent_logits[0].softmax(-1)
    k = int(probs.argmax())
    name = INTENT_CLASSES[k]
    cls_ok = name not in EXTRA_INTENT_CLASSES and float(probs[k]) >= cls_threshold
    cls_slots = {}
    if name in SLOTS:
        slot_name, values = SLOTS[name]
        cls_slots = {slot_name: values[int(head.slot_logits[slot_head_name(name)][0].argmax())]}
    return d, name, float(probs[k]), cls_ok, cls_slots


def hybrid_decode_all(
    ctc_model,
    cls_model,
    feature_extractor: LogMelFeatureExtractor,
    waveform: torch.Tensor,
    grammar,
    *,
    ctc_threshold: float = -0.1,
    margin: float | None = 4.0,
    cls_threshold: float,
    beam_width: int = 50,
    policies: tuple[str, ...] = ("fallback", "agree"),
) -> dict[str, HybridDecision]:
    """One encoder pass (two if the CTC and head checkpoints differ) and one beam search; every policy from them."""
    d, name, _, cls_ok, cls_slots = _forward(ctc_model, cls_model, feature_extractor, waveform, grammar, ctc_threshold=ctc_threshold,
                                             margin=margin, cls_threshold=cls_threshold, beam_width=beam_width)
    return {p: _combine(d, name, cls_ok, cls_slots, p) for p in policies}


def hybrid_decode(ctc_model, cls_model, feature_extractor, waveform, grammar, *, policy: str = "fallback", **kw) -> HybridDecision:
    return hybrid_decode_all(ctc_model, cls_model, feature_extractor, waveform, grammar, policies=(policy,), **kw)[policy]


class LazyCascade:
    """Policy `fallback` with the head model loaded and run only when the CTC rejects.

    `cls_loader` is called once, on the first rejected clip, and must return a model with `forward_heads`. If it is the
    CTC model itself no second pass is made. `n_cls_calls` counts how many clips needed the head model."""

    def __init__(self, ctc_model, cls_loader, feature_extractor, grammar, *, cls_threshold: float,
                 ctc_threshold: float = -0.1, margin: float | None = 4.0, beam_width: int = 50) -> None:
        self.ctc_model, self._loader, self.fe, self.grammar = ctc_model, cls_loader, feature_extractor, grammar
        self.cls_threshold, self.ctc_threshold, self.margin, self.beam = cls_threshold, ctc_threshold, margin, beam_width
        self._cls_model = None
        self.n_clips = self.n_cls_calls = 0

    @property
    def cls_loaded(self) -> bool:
        return self._cls_model is not None

    @torch.no_grad()
    def decode(self, waveform: torch.Tensor) -> HybridDecision:
        self.n_clips += 1
        features = self.fe(waveform).unsqueeze(0)
        d = decode(self.ctc_model(features).log_softmax(-1)[0].numpy(), self.grammar, self.ctc_threshold,
                   beam_width=self.beam, required_command_margin=self.margin)
        if d.intent is not None:
            return HybridDecision(d.intent, {k: str(v) for k, v in d.slots.items()}, "ctc")
        if self._cls_model is None:
            self._cls_model = self._loader()
        self.n_cls_calls += 1
        head = self._cls_model.forward_heads(features)
        probs = head.intent_logits[0].softmax(-1)
        k = int(probs.argmax())
        name = INTENT_CLASSES[k]
        if name in EXTRA_INTENT_CLASSES or float(probs[k]) < self.cls_threshold:
            return HybridDecision(None, {}, None)
        slots = {}
        if name in SLOTS:
            slot_name, values = SLOTS[name]
            slots = {slot_name: values[int(head.slot_logits[slot_head_name(name)][0].argmax())]}
        return HybridDecision(name, slots, "cls")


def _answer_ok(intent: str | None, slots: dict, label: str, slot_value: str) -> bool:
    if intent != label:
        return False
    return not slot_value or slot_value.lower() in [str(v).lower() for v in slots.values()]


class HybridDecoder:
    """Both models resident (the serving mode). `decode` returns the decision and a trace of what each component said;
    `stats` counts which path answered, and, when the caller supplies the truth to `decode`, which component was right:
    `ctc_right`, `cls_right`, `both`, `only_ctc`, `only_cls`, `neither`. `ctc_model` and `cls_model` may be the same object."""

    def __init__(self, ctc_model, cls_model, feature_extractor, grammar, *, cls_threshold: float, ctc_threshold: float = -0.1,
                 margin: float | None = 4.0, beam_width: int = 50, policy: str = "fallback", ctc_name: str = "ctc", cls_name: str = "cls") -> None:
        self.ctc_model, self.cls_model, self.fe, self.grammar = ctc_model, cls_model, feature_extractor, grammar
        self.kw = dict(ctc_threshold=ctc_threshold, margin=margin, cls_threshold=cls_threshold, beam_width=beam_width)
        self.policy, self.ctc_name, self.cls_name = policy, ctc_name, cls_name
        self.stats: dict[str, int] = {}

    def _bump(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def decode(self, waveform: torch.Tensor, truth: tuple[str, str] | None = None) -> tuple[HybridDecision, dict]:
        d, name, conf, cls_ok, cls_slots = _forward(self.ctc_model, self.cls_model, self.fe, waveform, self.grammar, **self.kw)
        decision = _combine(d, name, cls_ok, cls_slots, self.policy)
        ctc_slots = {k: str(v) for k, v in d.slots.items()}
        trace = {"ctc": self.ctc_name, "cls": self.cls_name, "ctc_intent": d.intent, "ctc_confidence": d.confidence if d.intent else None,
                 "cls_intent": name, "cls_confidence": conf, "cls_accepted": cls_ok, "agree": d.intent == name, "source": decision.source}
        self._bump("clips")
        self._bump(f"answered_by_{decision.source or 'none'}")
        if d.intent is not None and cls_ok:
            self._bump("both_accepted_agree" if d.intent == name else "both_accepted_disagree")
        if truth is not None:
            ctc_r = _answer_ok(d.intent, ctc_slots, *truth)
            cls_r = cls_ok and _answer_ok(name, cls_slots, *truth)
            trace["ctc_right"], trace["cls_right"] = ctc_r, cls_r
            self._bump("ctc_right") if ctc_r else None
            self._bump("cls_right") if cls_r else None
            self._bump("both" if ctc_r and cls_r else "only_ctc" if ctc_r else "only_cls" if cls_r else "neither")
            self._bump("final_right") if _answer_ok(decision.intent, decision.slots, *truth) else None
        return decision, trace


class _HeadsOut:
    def __init__(self, intent_logits: torch.Tensor, slot_logits: dict) -> None:
        self.intent_logits, self.slot_logits = intent_logits, slot_logits


class OnnxCtc:
    """ONNX CTC model (`vcm_model.*.onnx`: features -> logits) usable wherever `hybrid` calls `ctc_model(features)`."""

    def __init__(self, path, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.session.run(None, {"features": features.numpy().astype("float32")})[0])


class OnnxHeads:
    """ONNX heads-only model (`vcm_heads.*.onnx` from `vcm.export_onnx --heads-only`) usable as `cls_model` in `hybrid`."""

    def __init__(self, path, threads: int = 1) -> None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.names = [o.name for o in self.session.get_outputs()]

    def forward_heads(self, features: torch.Tensor) -> _HeadsOut:
        outs = dict(zip(self.names, self.session.run(None, {"features": features.numpy().astype("float32")})))
        slots = {slot_head_name(n[len("slot_"):]): torch.from_numpy(v) for n, v in outs.items() if n.startswith("slot_")}
        return _HeadsOut(torch.from_numpy(outs["intent_logits"]), slots)


def load_hybrid_part(path, role: str):
    """`.onnx` -> `OnnxCtc` / `OnnxHeads` (by role); anything else -> the PyTorch checkpoint."""
    from pathlib import Path

    if Path(path).suffix == ".onnx":
        return OnnxCtc(path) if role == "ctc" else OnnxHeads(path)
    from me2_voicegen.vcm.pipeline import load_checkpoint

    return load_checkpoint(path, device="cpu", weights_only=True)[0]


_FEATURES = None


@torch.no_grad()
def classifier_result(heads, waveform, threshold: float):
    """Classifier-head answer for one window as a `DecodeResult` (the streaming policy's override), or None when the head
    rejects (top class is unknown/silence or below `threshold`). `heads`: an `OnnxHeads` or a heads checkpoint."""
    import math

    import numpy as np

    from me2_voicegen.vcm.decoder import DecodeResult

    global _FEATURES
    if _FEATURES is None:
        _FEATURES = LogMelFeatureExtractor()
    wav = torch.as_tensor(np.asarray(waveform, dtype=np.float32))
    head = heads.forward_heads(_FEATURES(wav).unsqueeze(0))
    probs = head.intent_logits[0].softmax(-1)
    k = int(probs.argmax())
    name = INTENT_CLASSES[k]
    if name in EXTRA_INTENT_CLASSES or float(probs[k]) < threshold:
        return None
    slots = {}
    if name in SLOTS:
        slot_name, values = SLOTS[name]
        slots = {slot_name: values[int(head.slot_logits[slot_head_name(name)][0].argmax())]}
    return DecodeResult(intent=name, slots=slots, text="[classifier]", confidence=math.log(float(probs[k])),
                        no_match=False, out_of_grammar_gap=0.0)
