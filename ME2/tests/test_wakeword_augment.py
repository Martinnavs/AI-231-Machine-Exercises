import torch

from me2_voicegen.wakeword.augment import center_window, shift_waveform


def _nonzero_bounds(waveform: torch.Tensor) -> tuple[int, int]:
    nonzero = (waveform != 0).nonzero(as_tuple=True)[0]
    return int(nonzero.min()), int(nonzero.max())


def test_shift_waveform_pads_short_input_to_window_length():
    generator = torch.Generator().manual_seed(0)
    waveform = torch.ones(100)
    out = shift_waveform(waveform, window_samples=1000, generator=generator)
    assert out.shape[-1] == 1000
    start, end = _nonzero_bounds(out)
    assert end - start + 1 == 100
    assert 0 <= start <= 900


def test_shift_waveform_never_moves_content_outside_window():
    generator = torch.Generator().manual_seed(0)
    waveform = torch.ones(50)
    for _ in range(20):
        out = shift_waveform(waveform, window_samples=200, generator=generator)
        start, end = _nonzero_bounds(out)
        assert 0 <= start
        assert end < 200


def test_shift_waveform_randomizes_position_across_calls():
    generator = torch.Generator().manual_seed(0)
    waveform = torch.ones(10)
    starts = set()
    for _ in range(30):
        out = shift_waveform(waveform, window_samples=100, generator=generator)
        start, _end = _nonzero_bounds(out)
        starts.add(start)
    assert len(starts) > 1  # not deterministically the same offset every call


def test_shift_waveform_random_crops_when_longer_than_window():
    generator = torch.Generator().manual_seed(0)
    waveform = torch.arange(500, dtype=torch.float32)
    out = shift_waveform(waveform, window_samples=100, generator=generator)
    assert out.shape[-1] == 100
    # every value in the crop must be a contiguous slice of the source
    diffs = out[1:] - out[:-1]
    assert torch.all(diffs == 1.0)


def test_center_window_is_deterministic_and_pads_short_input():
    waveform = torch.ones(10)
    out1 = center_window(waveform, window_samples=100)
    out2 = center_window(waveform, window_samples=100)
    assert torch.equal(out1, out2)
    assert out1.shape[-1] == 100
    start, end = _nonzero_bounds(out1)
    assert start == 45  # (100 - 10) // 2
    assert end - start + 1 == 10


def test_center_window_crops_longer_input_to_middle():
    waveform = torch.arange(100, dtype=torch.float32)
    out = center_window(waveform, window_samples=20)
    assert out.shape[-1] == 20
    assert out[0].item() == 40  # (100 - 20) // 2
