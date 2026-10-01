"""MatchboxNet-style 1D time-channel-separable-conv CTC acoustic model.

Architecture: a prologue 1D conv, N residual time-channel-separable (TCS)
conv blocks (depthwise conv along time + pointwise 1x1 conv across
channels, per block, with a residual add and BatchNorm/ReLU), an epilogue
pair of convs, and a final linear projection to the 29-token CTC alphabet
(`vcm.alphabet.ALPHABET_SIZE`, per docs/VCM-CONTRACT.md section 1).

Input: `(B, 40, T)` log-mel features (docs/VCM-CONTRACT.md section 5).
Output: `(B, T, 29)` per-frame logits -- `nn.CTCLoss` wants
`(T, B, C)` log-probs, so callers permute/log_softmax themselves (kept
out of this module so it stays reusable for both training and inference,
where the two call sites want different shapes/dtypes around the same
core logits).

This model never changes the time dimension (`output_lengths` is the
identity); the strided `vcm.quartznet.QuartzNetCTC` does, so CTC callers must
always use `model.output_lengths(input_lengths)`.

Four presets:
    - default: sized for this toy's ~32 minutes of training audio,
      roughly 250k-400k params (this task's own engineering call, not
      dictated by the original spec).
    - "spec-scale": the original spec's stated 1.2M-2.5M parameter
      range, wider channels, for size-budget reporting -- NOT trained
      this pass (see ticket 04's Non-Goals).
    - "optionc": a mid-scale config (~754k params, still under 1M) added
      to test whether a larger-than-`default` capacity measurably improves
      decode accuracy on a given manifest -- deeper (4 TCS blocks vs 3) and
      wider (112 channels vs 72) than `default`, but well short of
      `spec-scale`'s width. Trained the same way as `default`
      (`--preset optionc`); not part of the original spec.
    - "optiond": the normal-scale config (~1.01M params), with 5 TCS
      blocks, 128 channels, a 64-channel prologue, and a 224-channel
      epilogue. It is trained through the same CLI and Make target as the
      other presets (`--preset optiond`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn

from me2_voicegen.vcm.alphabet import ALPHABET_SIZE
from me2_voicegen.vcm.quartznet import QUARTZNET5X3_CONFIG, QuartzNetConfig, QuartzNetCTC

MODEL_TYPE_KEY = "model_type"
LEGACY_MODEL_TYPE = "matchboxnet"


@dataclass
class MatchboxNetConfig:
    n_mels: int = 40
    n_blocks: int = 3
    channels: int = 96
    kernel_sizes: list[int] = field(default_factory=lambda: [11, 13, 15])
    prologue_channels: int = 64
    epilogue_channels: int = 128
    alphabet_size: int = ALPHABET_SIZE

    def __post_init__(self) -> None:
        if len(self.kernel_sizes) != self.n_blocks:
            if len(self.kernel_sizes) == 1:
                self.kernel_sizes = list(self.kernel_sizes) * self.n_blocks
            else:
                raise ValueError(
                    f"kernel_sizes has {len(self.kernel_sizes)} entries, "
                    f"expected {self.n_blocks} (one per TCS block)"
                )


DEFAULT_CONFIG = MatchboxNetConfig(
    n_mels=40,
    n_blocks=3,
    channels=72,
    kernel_sizes=[11, 13, 15],
    prologue_channels=48,
    epilogue_channels=96,
)
"""~254k params -- in the 250k-400k toy-scale target this task's own
engineering call chose (docs/VCM-CONTRACT.md doesn't dictate this; the
spec's 1.2M-2.5M figure is `SPEC_SCALE_CONFIG` below instead)."""

SPEC_SCALE_CONFIG = MatchboxNetConfig(
    n_mels=40,
    n_blocks=5,
    channels=192,
    kernel_sizes=[11, 13, 15, 17, 19],
    prologue_channels=96,
    epilogue_channels=320,
)
"""~2.14M params -- inside the original spec's stated 1.2M-2.5M range.
Reportable via `--preset spec-scale` but NOT trained this pass (ticket
04's Non-Goals)."""

OPTIONC_CONFIG = MatchboxNetConfig(
    n_mels=40,
    n_blocks=4,
    channels=112,
    kernel_sizes=[11, 13, 15, 17],
    prologue_channels=64,
    epilogue_channels=192,
)
"""~754k params (~3x DEFAULT_CONFIG, well under 1M) -- a mid-capacity
config for comparing decode accuracy against `default` on the same
manifest, to test whether more parameters actually help. Trained the same
way as `default` (`--preset optionc`)."""

OPTIOND_CONFIG = MatchboxNetConfig(
    n_mels=40,
    n_blocks=5,
    channels=128,
    kernel_sizes=[11, 13, 15, 17, 19],
    prologue_channels=64,
    epilogue_channels=224,
)
"""~1.01M params -- the normal-scale MatchboxNetCTC config."""

OPTIOND_WIDE_CONFIG = MatchboxNetConfig(
    n_mels=40,
    n_blocks=5,
    channels=128,
    kernel_sizes=[61, 61, 61, 61, 61],
    prologue_channels=64,
    epilogue_channels=224,
)
"""Ablation control (not a shipping preset): `optiond` with block kernels widened
to 61 (receptive field ~3 s vs ~1.1 s), epilogue unchanged. ~1.036M params,
26k over the optiond budget, so its INT8 export exceeds the size gate."""

PRESETS: dict[str, MatchboxNetConfig | QuartzNetConfig] = {
    "default": DEFAULT_CONFIG,
    "spec-scale": SPEC_SCALE_CONFIG,
    "optionc": OPTIONC_CONFIG,
    "optiond": OPTIOND_CONFIG,
    "quartznet5x3": QUARTZNET5X3_CONFIG,
    "optiond-wide": OPTIOND_WIDE_CONFIG,
    "quartznet5x3-s1": dataclasses.replace(QUARTZNET5X3_CONFIG, time_stride=1),
}


class CTCAcousticModel(Protocol):
    """The seam every VCM acoustic model exposes to train/export/inference."""

    def forward(self, features: torch.Tensor) -> torch.Tensor: ...

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor: ...


class TCSConvBlock(nn.Module):
    """One residual time-channel-separable conv block.

    Depthwise (per-channel) 1D conv along time, then a pointwise 1x1 conv
    mixing channels, each followed by BatchNorm; residual add (with a 1x1
    projection when channel counts differ) then ReLU.
    """

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
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.residual_bn = (
            nn.BatchNorm1d(out_channels) if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_bn(self.residual_proj(x))
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        out = out + residual
        return self.act(out)


class MatchboxNetCTC(nn.Module):
    """Prologue conv -> N residual TCS blocks -> epilogue convs -> linear
    -> per-frame logits over the 29-token CTC alphabet."""

    def __init__(self, config: MatchboxNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        self.prologue = nn.Sequential(
            nn.Conv1d(
                self.config.n_mels,
                self.config.prologue_channels,
                kernel_size=11,
                padding=5,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.prologue_channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = self.config.prologue_channels
        for kernel_size in self.config.kernel_sizes:
            blocks.append(TCSConvBlock(in_ch, self.config.channels, kernel_size))
            in_ch = self.config.channels
        self.blocks = nn.Sequential(*blocks)

        self.epilogue = nn.Sequential(
            nn.Conv1d(in_ch, self.config.epilogue_channels, kernel_size=29, padding=14, bias=False),
            nn.BatchNorm1d(self.config.epilogue_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                self.config.epilogue_channels,
                self.config.epilogue_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.config.epilogue_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(self.config.epilogue_channels, self.config.alphabet_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """`(B, n_mels, T)` -> `(B, T, alphabet_size)` logits."""
        x = self.prologue(features)
        x = self.blocks(x)
        x = self.epilogue(x)
        x = x.transpose(1, 2)
        return self.classifier(x)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Every conv is stride 1 with length-preserving padding: identity."""
        return input_lengths.clone()


ARCHITECTURES: dict[str, tuple[type, type[nn.Module]]] = {
    LEGACY_MODEL_TYPE: (MatchboxNetConfig, MatchboxNetCTC),
    "quartznet": (QuartzNetConfig, QuartzNetCTC),
}


def model_type_for_config(config: MatchboxNetConfig | QuartzNetConfig) -> str:
    for name, (config_cls, _) in ARCHITECTURES.items():
        if isinstance(config, config_cls):
            return name
    raise ValueError(f"unsupported model config type {type(config).__name__}")


def build_model_from_config(config: dict, model_type: str | None = None) -> nn.Module:
    """Build a model from a checkpoint's `config` dict and `model_type` key.

    `model_type=None` (checkpoints written before the key existed) means
    MatchboxNet. The class comes from the closed `ARCHITECTURES` table, never
    from checkpoint-supplied names.
    """
    key = LEGACY_MODEL_TYPE if model_type is None else model_type
    if key not in ARCHITECTURES:
        raise ValueError(f"unknown model_type {key!r}, expected one of {sorted(ARCHITECTURES)}")
    config_cls, model_cls = ARCHITECTURES[key]
    return model_cls(config_cls(**config))


def build_model(preset: str = "default") -> nn.Module:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}, expected one of {sorted(PRESETS)}")
    config = PRESETS[preset]
    _, model_cls = ARCHITECTURES[model_type_for_config(config)]
    return model_cls(config)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimated_int8_bytes(model: nn.Module) -> int:
    """Estimate a post-training-INT8-quantized checkpoint's size in bytes:
    1 byte/param plus a per-tensor float32 scale (4 bytes) for every
    parameter tensor that would carry its own quantization scale."""
    total = 0
    for p in model.parameters():
        total += p.numel()
    n_tensors = sum(1 for _ in model.parameters())
    return total + n_tensors * 4


if __name__ == "__main__":
    for name in PRESETS:
        model = build_model(name)
        n_params = param_count(model)
        n_bytes = estimated_int8_bytes(model)
        print(f"{name}: {n_params:,} params, ~{n_bytes / 1024:.1f} KiB estimated INT8")
