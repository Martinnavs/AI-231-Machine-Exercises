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
