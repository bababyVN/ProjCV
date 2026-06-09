# =============================================================
#  swin_encoder.py — SwinFAN Encoder
#
#  Implements a Swin Transformer (swin_t) backbone with a
#  lightweight CNN stem and multi-scale feature extraction.
#
#  Architecture
#  ─────────────────────────────────────────────────────────────
#  Input: (B, 3, H, W)
#
#  Stage  | Module             | Output (B, C, H/s, W/s) | Stride
#  -------|--------------------|--------------------------|-------
#  stem   | Conv3×3 BN ReLU    | (B,   3, H,      W)     |  ×1
#  s2     | Conv3×3 stride-2   | (B,  48, H/2,   W/2)    |  ×2
#  s4     | Swin stage 1       | (B,  96, H/4,   W/4)    |  ×4
#  s8     | Swin stage 2       | (B, 192, H/8,   W/8)    |  ×8
#  s16    | Swin stage 3       | (B, 384, H/16, W/16)    | ×16
#  s32    | Swin stage 4       | (B, 768, H/32, W/32)    | ×32
#
#  Returns: [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck_s32]
#  encoder_channels = [3, 48, 96, 192, 384, 768]
#
#  Note: Swin-T outputs tensors in (B, H, W, C) format;
#        we permute to (B, C, H, W) for the decoder.
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from config import CFG


class SwinEncoder(nn.Module):
    """
    SwinFAN Encoder: Swin-T backbone with a CNN Stem for high-resolution skips.

    Architecture
    ------------
    We augment the 4-stage Swin-T with two lightweight CNN stages (stride 1
    and stride 2) so that the UNet decoder has enough resolution levels to
    reconstruct fine-grained segmentation maps at full resolution.

    The Swin Transformer uses hierarchical shifted-window self-attention to
    capture long-range dependencies efficiently across multiple scales — the
    defining property of the SwinFAN architecture.

    Feature Map Summary (input 512×512):
        skip_s1  : (B,   3, 512, 512)  — pass-through input (stride 1)
        skip_s2  : (B,  48, 256, 256)  — CNN stem stride-2 projection
        skip_s4  : (B,  96, 128, 128)  — Swin stage 1 (patch embed + 2 blocks)
        skip_s8  : (B, 192,  64,  64)  — Swin stage 2 (patch merge + 2 blocks)
        skip_s16 : (B, 384,  32,  32)  — Swin stage 3 (patch merge + 6 blocks)
        bottleneck:(B, 768,  16,  16)  — Swin stage 4 (patch merge + 2 blocks)

    Returns
    -------
        List[Tensor] — ordered highest-resolution to lowest-resolution.
        encoder_channels attribute consumed by the decoder.
    """

    def __init__(self, in_channels: int = 3, config: dict = CFG):
        super().__init__()
        self.config = config

        # ── CNN Stem ─────────────────────────────────────────────────────────
        # skip_s1: stride-1 pass-through of the normalised input image.
        # Used as the highest-resolution skip to restore fine pixel details.
        # We just pass the raw input (after ImageNet normalisation) directly.
        # No extra convolution needed — 3 channels is enough at stride 1.

        # skip_s2: stride-2 lightweight projection.
        # Bridges the gap between stride-1 raw pixels and stride-4 Swin output.
        self.stem_s2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ── Swin-T Backbone ──────────────────────────────────────────────────
        # Load Swin-T with ImageNet pretrained weights.
        # We extract 4 hierarchical feature scales using create_feature_extractor.
        try:
            backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            print("[encoder] Swin-T: ImageNet pretrained weights loaded.")
        except Exception:
            backbone = swin_t(weights=None)
            print("[encoder] WARNING: Swin-T pretrained weights unavailable — "
                  "using random initialisation.")

        # Extract the 4 Swin stages via the end of each transformer block group.
        # features.1 → stage 1 output (B, H/4, W/4, 96)
        # features.3 → stage 2 output (B, H/8, W/8, 192)
        # features.5 → stage 3 output (B, H/16, W/16, 384)
        # features.7 → stage 4 output (B, H/32, W/32, 768)
        self.swin = create_feature_extractor(backbone, return_nodes={
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        })

        # ── Channel Specification ─────────────────────────────────────────────
        # Consumed by the decoder to configure its blocks.
        # Order: [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck_s32]
        self.encoder_channels = [in_channels, 48, 96, 192, 384, 768]

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x : (B, 3, H, W) — ImageNet-normalised float tensor.

        Returns:
            List of 6 feature tensors ordered highest → lowest resolution:
                [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck_s32]
        """
        # Skip 1 — stride 1: raw input pixels (B, 3, H, W)
        skip_s1 = x

        # Skip 2 — stride 2: lightweight CNN projection (B, 48, H/2, W/2)
        skip_s2 = self.stem_s2(x)

        # Swin Transformer multi-scale features
        # Output tensors are in (B, H, W, C) layout from Swin-T; we permute.
        swin_feats = self.swin(x)

        # Skip 3 — stride 4:  (B, 96, H/4, W/4)
        skip_s4 = swin_feats["stage1"].permute(0, 3, 1, 2).contiguous()

        # Skip 4 — stride 8:  (B, 192, H/8, W/8)
        skip_s8 = swin_feats["stage2"].permute(0, 3, 1, 2).contiguous()

        # Skip 5 — stride 16: (B, 384, H/16, W/16)
        skip_s16 = swin_feats["stage3"].permute(0, 3, 1, 2).contiguous()

        # Bottleneck — stride 32: (B, 768, H/32, W/32)
        bottleneck = swin_feats["stage4"].permute(0, 3, 1, 2).contiguous()

        return [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck]


# ─────────────────────────────────────────────────────────────
# Quick shape test — run `python encoder/swin_encoder.py`
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("swin_encoder.py — Running shape test...")
    encoder = SwinEncoder(in_channels=3, config=CFG)
    dummy   = torch.randn(2, 3, 512, 512)
    feats   = encoder(dummy)

    names = ["skip_s1 (×1)", "skip_s2 (×2)", "skip_s4 (×4)",
             "skip_s8 (×8)", "skip_s16 (×16)", "bottleneck (×32)"]
    for name, feat in zip(names, feats):
        print(f"  {name:20s} → {tuple(feat.shape)}")

    expected_channels = [3, 48, 96, 192, 384, 768]
    for feat, exp_ch in zip(feats, expected_channels):
        assert feat.shape[1] == exp_ch, \
            f"Channel mismatch! Expected {exp_ch}, got {feat.shape[1]}"
    print("Shape test PASSED")

