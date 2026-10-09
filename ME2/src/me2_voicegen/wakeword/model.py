"""Depthwise-separable CNN (DS-CNN) keyword-spotting classifier for the
single wakeword "computer" (feature `wakeword-dscnn`,
feature-engineering/wakeword-dscnn/SPEC.md).

Architecture: a prologue standard 1D conv, N depthwise-separable conv
blocks (depthwise conv along time + pointwise 1x1 conv across channels,
each with BatchNorm/ReLU -- no residual add, unlike `vcm.model`'s TCS
blocks: canonical DS-CNN (Zhang et al., "Hello Edge") has none, and this
is a 3-way clip classifier, not a per-frame CTC model, so there is no
frame-alignment reason to add one here), a global-average-pool over time,
and a linear classifier to 3 classes.

Input: `(B, 40, T)` log-mel features (`common.features.LogMelFeatureExtractor`,
reused unmodified -- SPEC.md Assumptions). Output: `(B, 3)` logits over
`LABELS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

LABELS: tuple[str, str, str] = ("_wakeword_", "_unknown_", "_silence_")
LABEL_TO_ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}


@dataclass
class DSCNNConfig:
    n_mels: int = 40
    n_classes: int = 3
    n_blocks: int = 4
    channels: int = 64
    kernel_sizes: list[int] = field(default_factory=lambda: [10, 10, 10, 10])
    prologue_channels: int = 32

    def __post_init__(self) -> None:
        if len(self.kernel_sizes) != self.n_blocks:
            if len(self.kernel_sizes) == 1:
                self.kernel_sizes = list(self.kernel_sizes) * self.n_blocks
            else:
                raise ValueError(
                    f"kernel_sizes has {len(self.kernel_sizes)} entries, "
                    f"expected {self.n_blocks} (one per DS-conv block)"
                )


DEFAULT_CONFIG = DSCNNConfig(
    n_mels=40,
    n_classes=3,
    n_blocks=4,
    channels=64,
    kernel_sizes=[10, 10, 10, 10],
    prologue_channels=32,
)
"""~50k params -- small keyword-spotting scale, in the spirit of the
handoff doc's MLPerf Tiny DS-CNN reference (38.6-52.5 KB INT8), though not
held to that exact figure: this is a 3-way clip classifier over this
project's own dataset, not the 12-class Speech Commands benchmark the
MLPerf Tiny numbers were measured on."""

PRESETS: dict[str, DSCNNConfig] = {
    "default": DEFAULT_CONFIG,
}


class DSConvBlock(nn.Module):
    """One depthwise-separable conv block: depthwise conv along time, then
    a pointwise 1x1 conv mixing channels, each followed by BatchNorm/ReLU.
    No residual add (canonical DS-CNN, unlike `vcm.model.TCSConvBlock`)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.depthwise_bn(self.depthwise(x)))
        x = self.act(self.pointwise_bn(self.pointwise(x)))
        return x


class DSCNN(nn.Module):
    """Prologue conv -> N depthwise-separable conv blocks -> global average
    pool over time -> linear classifier."""

    def __init__(self, config: DSCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        self.prologue = nn.Sequential(
            nn.Conv1d(
                self.config.n_mels,
                self.config.prologue_channels,
                kernel_size=self.config.kernel_sizes[0],
                padding=self.config.kernel_sizes[0] // 2,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.prologue_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = self.config.prologue_channels
        for kernel_size in self.config.kernel_sizes:
            blocks.append(DSConvBlock(in_ch, self.config.channels, kernel_size))
            in_ch = self.config.channels
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.config.channels, self.config.n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, n_mels, T)` -> `(B, n_classes)` logits."""
        x = self.prologue(features)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def build_model(preset: str = "default") -> DSCNN:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, expected one of {sorted(PRESETS)}")
    return DSCNN(PRESETS[preset])


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimated_int8_bytes(model: nn.Module) -> int:
    """Estimate a post-training-INT8-quantized checkpoint's size in bytes:
    1 byte/param plus a per-tensor float32 scale (4 bytes) for every
    parameter tensor that would carry its own quantization scale (mirrors
    `vcm.model.estimated_int8_bytes`)."""
    total = sum(p.numel() for p in model.parameters())
    n_tensors = sum(1 for _ in model.parameters())
    return total + n_tensors * 4


if __name__ == "__main__":
    for name in PRESETS:
        model = build_model(name)
        n_params = param_count(model)
        n_bytes = estimated_int8_bytes(model)
        print(f"{name}: {n_params:,} params, ~{n_bytes / 1024:.1f} KiB estimated INT8")
