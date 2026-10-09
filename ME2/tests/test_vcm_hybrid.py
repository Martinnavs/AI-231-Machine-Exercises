import torch
from me2_voicegen.common.features import LogMelFeatureExtractor
from me2_voicegen.vcm.hybrid import hybrid_decode_all
from me2_voicegen.vcm.model import build_model
from me2_voicegen.vcm.optionb.grammar import OPTIONB_GRAMMAR


def test_hybrid_policies_on_untrained_heads_model():
    torch.manual_seed(0)
    m = build_model("quartznet5x3-heads").eval()
    wav = torch.randn(16000 * 2) * 0.01
    ex = LogMelFeatureExtractor()
    hi = hybrid_decode_all(m, m, ex, wav, OPTIONB_GRAMMAR, cls_threshold=1.01)  # classifier can never accept
    assert hi["fallback"].source != "cls" and hi["agree"].intent is None
    lo = hybrid_decode_all(m, m, ex, wav, OPTIONB_GRAMMAR, cls_threshold=0.0)  # classifier always accepts unless EXTRA class
    assert set(lo) == {"fallback", "agree"}
    for d in lo.values():
        assert d.source in ("ctc", "cls", None)


def test_lazy_cascade_loads_head_model_only_when_ctc_rejects():
    from me2_voicegen.vcm.hybrid import LazyCascade
    torch.manual_seed(0)
    m = build_model("quartznet5x3-heads").eval()
    calls = []
    lc = LazyCascade(m, lambda: calls.append(1) or m, LogMelFeatureExtractor(), OPTIONB_GRAMMAR, cls_threshold=0.0)
    wav = torch.randn(16000 * 2) * 0.01
    dec = lc.decode(wav)  # untrained model: CTC rejects noise, so the head model is needed
    ref = hybrid_decode_all(m, m, LogMelFeatureExtractor(), wav, OPTIONB_GRAMMAR, cls_threshold=0.0)["fallback"]
    assert (dec.intent, dec.source) == (ref.intent, ref.source)
    assert lc.cls_loaded == (dec.source != "ctc") and len(calls) <= 1


def test_hybrid_decoder_tracks_components():
    from me2_voicegen.vcm.hybrid import HybridDecoder
    torch.manual_seed(0)
    m = build_model("quartznet5x3-heads").eval()
    hd = HybridDecoder(m, m, LogMelFeatureExtractor(), OPTIONB_GRAMMAR, cls_threshold=0.0, ctc_name="a", cls_name="b")
    dec, tr = hd.decode(torch.randn(16000 * 2) * 0.01, truth=("PLAY_MUSIC", ""))
    assert hd.stats["clips"] == 1 and tr["ctc"] == "a" and tr["cls"] == "b"
    assert sum(hd.stats.get(k, 0) for k in ("both", "only_ctc", "only_cls", "neither")) == 1


def test_heads_only_onnx_export_matches_pytorch(tmp_path):
    import numpy as np
    from me2_voicegen.vcm.export_onnx import export_heads_fp32
    from me2_voicegen.vcm.hybrid import OnnxHeads
    from me2_voicegen.vcm.semantic_labels import SLOT_INTENTS, slot_head_name

    torch.manual_seed(0)
    m = build_model("quartznet5x3-heads").eval()
    path = export_heads_fp32(m, tmp_path / "heads.onnx")
    feats = torch.randn(1, 40, 180)  # a different length from the export's dummy input: time axis is dynamic
    with torch.no_grad():
        ref = m.forward_heads(feats)
    got = OnnxHeads(path).forward_heads(feats)
    np.testing.assert_allclose(got.intent_logits.numpy(), ref.intent_logits.numpy(), atol=1e-4)
    for i in SLOT_INTENTS:
        k = slot_head_name(i)
        np.testing.assert_allclose(got.slot_logits[k].numpy(), ref.slot_logits[k].numpy(), atol=1e-4)


def test_classifier_result_slot_threshold_rejects_unsure_slots_only_for_slotted_intents():
    from me2_voicegen.vcm.hybrid import classifier_result
    from me2_voicegen.vcm.semantic_labels import INTENT_CLASSES, SLOTS, slot_head_name

    class _Heads:
        def __init__(self, intent: str, slot_logit_gap: float) -> None:
            self.intent, self.gap = intent, slot_logit_gap

        def forward_heads(self, features):
            il = torch.zeros(1, len(INTENT_CLASSES))
            il[0, INTENT_CLASSES.index(self.intent)] = 20.0
            sl = {}
            for name in SLOTS:
                logits = torch.zeros(1, len(SLOTS[name][1]))
                logits[0, 1] = self.gap
                sl[slot_head_name(name)] = logits
            return type("Out", (), {"intent_logits": il, "slot_logits": sl})()

    wav = torch.randn(16000).numpy() * 0.01
    sure, unsure = _Heads("TIMER", 12.0), _Heads("TIMER", 0.1)
    assert classifier_result(sure, wav, 0.5, slot_threshold=0.9).slots == {"DURATION": SLOTS["TIMER"][1][1]}
    assert classifier_result(unsure, wav, 0.5, slot_threshold=0.9) is None
    assert classifier_result(unsure, wav, 0.5, slot_threshold=0.0) is not None            # off by default
    assert classifier_result(_Heads("PLAY_MUSIC", 0.1), wav, 0.5, slot_threshold=0.9).intent == "PLAY_MUSIC"  # no slot to doubt
