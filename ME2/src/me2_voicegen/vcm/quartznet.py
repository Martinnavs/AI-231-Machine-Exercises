"""QuartzNet-5x3-tiny character-CTC acoustic model with temporal subsampling.

Same seam as `vcm.model.MatchboxNetCTC` -- `(B, 40, T)` log-mel features in,
`(B, T', 29)` logits out -- except T' may be smaller than T: a stride-2
depthwise prologue (and, for `time_stride=4`, a stride-2 first sub-module in
block 1) subsamples time. Callers must therefore use `output_lengths()` for
CTC input lengths instead of the feature lengths (docs/VCM-CONTRACT.md,
model-seam note).

Architecture (time-channel-separable throughout, odd kernels, "same"-style
padding): prologue depthwise(k, stride 2) + pointwise + BN + ReLU; 5 residual
blocks of `repeats` (3) depthwise/pointwise/BN sub-modules (ReLU + dropout
between them, residual 1x1 + BN added before the block's last ReLU); an
epilogue depthwise + pointwise + 1x1; then a linear head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from me2_voicegen.vcm.alphabet import ALPHABET_SIZE


@dataclass
class QuartzNetConfig:
    n_mels: int = 40
    n_blocks: int = 5
    repeats: int = 3
    channels: int = 192
    kernel_sizes: list[int] = field(default_factory=lambda: [7, 9, 11, 13, 15])
    prologue_kernel: int = 11
    epilogue_kernel: int = 15
    epilogue_channels: int = 256
    time_stride: int = 2  # 1 (ablation: no subsampling), 2 (default) or 4
    dropout: float = 0.0
    alphabet_size: int = ALPHABET_SIZE

    def __post_init__(self) -> None:
        if self.time_stride not in (1, 2, 4):
            raise ValueError(f"time_stride must be 1, 2 or 4, got {self.time_stride}")
        if len(self.kernel_sizes) != self.n_blocks:
            if len(self.kernel_sizes) == 1:
                self.kernel_sizes = list(self.kernel_sizes) * self.n_blocks
            else:
                raise ValueError(
                    f"kernel_sizes has {len(self.kernel_sizes)} entries, "
                    f"expected {self.n_blocks} (one per block)"
                )
        for k in [*self.kernel_sizes, self.prologue_kernel, self.epilogue_kernel]:
            if k % 2 != 1 or k < 1:
                raise ValueError(f"kernel sizes must be positive and odd, got {k}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.repeats < 1 or self.n_blocks < 1:
            raise ValueError("repeats and n_blocks must be >= 1")


QUARTZNET5X3_CONFIG = QuartzNetConfig()
"""Default preset, 911,189 params. Width 200 (978,053 params) fits the parameter
budget but its INT8 ONNX export is 1,062,125 B, over the 1,048,576 B gate;
192 exports to 974,006 B. Width is set by that gate, not the parameter cap."""


def _sep_conv(in_ch: int, out_ch: int, kernel: int, stride: int) -> nn.Sequential:
    """Depthwise (time) conv then pointwise 1x1 (channels), no norm/act."""
    return nn.Sequential(
        nn.Conv1d(
            in_ch, in_ch, kernel_size=kernel, stride=stride, padding=kernel // 2,
            groups=in_ch, bias=False,
        ),
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
    )


class _QuartzBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, repeats: int, dropout: float, first_stride: int) -> None:
        super().__init__()
        self.subs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(repeats):
            self.subs.append(_sep_conv(channels, channels, kernel, first_stride if i == 0 else 1))
            self.norms.append(nn.BatchNorm1d(channels))
        self.residual = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, stride=first_stride, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        last = len(self.subs) - 1
        for i, (sub, norm) in enumerate(zip(self.subs, self.norms)):
            out = norm(sub(out))
            if i < last:
                out = self.drop(self.act(out))
        return self.drop(self.act(out + self.residual(x)))


class QuartzNetCTC(nn.Module):
    """`(B, n_mels, T)` -> `(B, T', alphabet_size)` logits, T' = ceil(T / total_stride)."""

    def __init__(self, config: QuartzNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or QUARTZNET5X3_CONFIG
        c = self.config
        self.total_stride = c.time_stride
        block1_stride = 2 if c.time_stride == 4 else 1

        prologue_stride = 1 if c.time_stride == 1 else 2
        self.prologue = nn.Sequential(
            _sep_conv(c.n_mels, c.channels, c.prologue_kernel, stride=prologue_stride),
            nn.BatchNorm1d(c.channels),
            nn.ReLU(),
            nn.Dropout(c.dropout),
        )
        self.blocks = nn.Sequential(
            *[
                _QuartzBlock(c.channels, k, c.repeats, c.dropout, block1_stride if i == 0 else 1)
                for i, k in enumerate(c.kernel_sizes)
            ]
        )
        self.epilogue = nn.Sequential(
            _sep_conv(c.channels, c.epilogue_channels, c.epilogue_kernel, stride=1),
            nn.BatchNorm1d(c.epilogue_channels),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Conv1d(c.epilogue_channels, c.epilogue_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(c.epilogue_channels),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(c.epilogue_channels, c.alphabet_size)

        # (kernel, stride, padding, dilation) of every length-changing conv on
        # the main path, in order; output_lengths() applies the conv formula.
        self._length_convs: list[tuple[int, int, int, int]] = [
            (c.prologue_kernel, prologue_stride, c.prologue_kernel // 2, 1)
        ]
        if block1_stride != 1:
            self._length_convs.append((c.kernel_sizes[0], block1_stride, c.kernel_sizes[0] // 2, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.prologue(features)
        x = self.blocks(x)
        x = self.epilogue(x)
        return self.classifier(x.transpose(1, 2))

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.to(torch.long)
        for k, s, p, d in self._length_convs:
            lengths = (lengths + 2 * p - d * (k - 1) - 1) // s + 1
        return lengths
