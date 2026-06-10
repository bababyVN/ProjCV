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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torchvision.models import (swin_t, Swin_T_Weights,
                                swin_b, Swin_B_Weights,
                                swin_v2_b, Swin_V2_B_Weights)
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


class SwinBaseEncoder(nn.Module):
    """
    SwinFAN-v2 Encoder: Swin-Base backbone with a CNN stem.

    Doubles the channel width compared to SwinEncoder (Swin-T),
    allowing the model to learn richer multi-scale representations
    for satellite imagery.

    Architecture
    ------------
    We augment the 4-stage Swin-B with the same two CNN stages
    (stride 1 and stride 2) as SwinEncoder to give the decoder
    enough resolution levels for fine-grained segmentation.

    Swin-B uses embed_dim=128 (vs. 96 for Swin-T), so all stages
    output double the channels:

    Feature Map Summary (input 512×512):
        skip_s1  : (B,    3, 512, 512)  — pass-through input (stride 1)
        skip_s2  : (B,   48, 256, 256)  — CNN stem stride-2 projection
        skip_s4  : (B,  128, 128, 128)  — Swin-B stage 1
        skip_s8  : (B,  256,  64,  64)  — Swin-B stage 2
        skip_s16 : (B,  512,  32,  32)  — Swin-B stage 3
        bottleneck:(B, 1024,  16,  16)  — Swin-B stage 4

    Returns
    -------
        List[Tensor] — ordered highest-resolution to lowest-resolution.
        encoder_channels attribute consumed by the decoder.
    """

    def __init__(self, in_channels: int = 3, config: dict = CFG):
        super().__init__()
        self.config = config

        # ── CNN Stem ─────────────────────────────────────────────────────────
        # Identical to SwinEncoder: stride-2 lightweight projection.
        self.stem_s2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ── Swin-B Backbone ──────────────────────────────────────────────────
        # Load Swin-B with ImageNet-1K pretrained weights.
        # Swin-B: embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32]
        try:
            backbone = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
            print("[encoder] Swin-B: ImageNet pretrained weights loaded.")
        except Exception:
            backbone = swin_b(weights=None)
            print("[encoder] WARNING: Swin-B pretrained weights unavailable — "
                  "using random initialisation.")

        # Extract the 4 Swin-B stages.
        # Swin-B output layout matches Swin-T structure; only channels differ:
        #   features.1 → stage 1 output (B, H/4,  W/4,  128)
        #   features.3 → stage 2 output (B, H/8,  W/8,  256)
        #   features.5 → stage 3 output (B, H/16, W/16, 512)
        #   features.7 → stage 4 output (B, H/32, W/32, 1024)
        self.swin = create_feature_extractor(backbone, return_nodes={
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        })

        # ── Channel Specification ─────────────────────────────────────────────
        # Consumed by the decoder to configure its blocks.
        # Order: [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck]
        self.encoder_channels = [in_channels, 48, 128, 256, 512, 1024]

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

        # Swin-B multi-scale features (tensors in B, H, W, C layout)
        swin_feats = self.swin(x)

        # Skip 3 — stride 4:  (B, 128, H/4, W/4)
        skip_s4 = swin_feats["stage1"].permute(0, 3, 1, 2).contiguous()

        # Skip 4 — stride 8:  (B, 256, H/8, W/8)
        skip_s8 = swin_feats["stage2"].permute(0, 3, 1, 2).contiguous()

        # Skip 5 — stride 16: (B, 512, H/16, W/16)
        skip_s16 = swin_feats["stage3"].permute(0, 3, 1, 2).contiguous()

        # Bottleneck — stride 32: (B, 1024, H/32, W/32)
        bottleneck = swin_feats["stage4"].permute(0, 3, 1, 2).contiguous()

        return [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck]


# ─────────────────────────────────────────────────────────────
# SwinV2BaseEncoder — SwinFAN-v3 (SatlasPretrain RS Pretrained)
# ─────────────────────────────────────────────────────────────
class SwinV2BaseEncoder(nn.Module):
    """
    SwinFAN-v3 Encoder: Swin-v2-Base backbone pretrained on
    SatlasPretrain aerial imagery (ICCV 2023).

    Backbone: torchvision.models.swin_v2_b
        - Swin Transformer V2 with log-spaced continuous position bias
          and scaled cosine attention — more robust at high resolutions.
        - Output channels are identical to Swin-v1-Base:
          128 → 256 → 512 → 1024 across 4 stages.

    Pretraining: SatlasPretrain Aerial_SwinB_SI
        - Pretrained on 302M remote sensing labels across 137 categories.
        - Aerial imagery at 0.5–2 m/pixel (same scale as DeepGlobe 50 cm).
        - Expected +3–5% mIoU vs ImageNet-1K pretrained Swin-B.
        - Published at ICCV 2023 by Allen AI (Apache-2.0 license).

    Weight loading:
        The checkpoint is downloaded once at runtime from HuggingFace
        (~503 MB) and cached at /kaggle/working/weights/ (or ~/.cache/satlas/).
        Keys are stripped of the "backbone." prefix and loaded with
        strict=False — absent classification head keys are silently skipped.

    Feature Map Summary (input 512x512):
        skip_s1  : (B,    3, 512, 512)  — pass-through input (stride 1)
        skip_s2  : (B,   48, 256, 256)  — CNN stem stride-2 projection
        skip_s4  : (B,  128, 128, 128)  — Swin-v2-B stage 1
        skip_s8  : (B,  256,  64,  64)  — Swin-v2-B stage 2
        skip_s16 : (B,  512,  32,  32)  — Swin-v2-B stage 3
        bottleneck:(B, 1024,  16,  16)  — Swin-v2-B stage 4

    encoder_channels = [3, 48, 128, 256, 512, 1024]
    (identical to SwinBaseEncoder — decoder needs no changes)
    """

    # HuggingFace direct-download URL for the Aerial_SwinB_SI checkpoint.
    # Apache-2.0 license. File size: ~503 MB.
    SATLAS_AERIAL_URL = (
        "https://huggingface.co/allenai/satlas-pretrain"
        "/resolve/main/aerial_swinb_si.pth?download=true"
    )

    def __init__(self, in_channels: int = 3, config: dict = CFG):
        super().__init__()
        self.config = config

        # ── CNN Stem ─────────────────────────────────────────────────────────
        # Identical to SwinBaseEncoder: lightweight stride-2 projection.
        self.stem_s2 = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # ── Swin-v2-B Backbone (no ImageNet weights — loads RS pretrain) ──────
        backbone = swin_v2_b(weights=None)

        # ── Load SatlasPretrain Aerial weights ────────────────────────────────
        self._load_satlas_weights(backbone)

        # ── Feature extractor: same node names as Swin-v1-B ──────────────────
        # Swin-v2-B and Swin-v1-B share the same torchvision feature hierarchy.
        #   features.1 -> stage 1 output (B, H/4,  W/4,  128)
        #   features.3 -> stage 2 output (B, H/8,  W/8,  256)
        #   features.5 -> stage 3 output (B, H/16, W/16, 512)
        #   features.7 -> stage 4 output (B, H/32, W/32, 1024)
        self.swin = create_feature_extractor(backbone, return_nodes={
            "features.1": "stage1",
            "features.3": "stage2",
            "features.5": "stage3",
            "features.7": "stage4",
        })

        # ── Channel Specification ─────────────────────────────────────────────
        self.encoder_channels = [in_channels, 48, 128, 256, 512, 1024]

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_cache_path() -> str:
        """
        Return a local path to cache the SatlasPretrain weights.

        Priority order:
          1. /kaggle/working/weights/  (Kaggle notebook working dir)
          2. ~/.cache/satlas/          (local dev / generic fallback)
        """
        import os
        candidates = [
            "/kaggle/working/weights",
            os.path.join(os.path.expanduser("~"), ".cache", "satlas"),
        ]
        for d in candidates:
            try:
                os.makedirs(d, exist_ok=True)
                return os.path.join(d, "aerial_swinb_si.pth")
            except OSError:
                continue
        # Last resort — current working directory
        return "aerial_swinb_si.pth"

    # ─────────────────────────────────────────────────────────────────────────
    def _load_satlas_weights(self, backbone: nn.Module) -> None:
        """
        Download (if needed) and load SatlasPretrain aerial weights into
        `backbone` (a torchvision swin_v2_b instance).

        The checkpoint stores backbone parameters under the "backbone."
        prefix. We strip that prefix and load with strict=False so the
        absent classification head keys are silently skipped.
        """
        import os

        cache_path = self._get_cache_path()

        # ── Step 1: Download if not cached ────────────────────────────────────
        if not os.path.isfile(cache_path):
            print("[encoder] Downloading SatlasPretrain aerial weights (~503 MB)...")
            print(f"[encoder]   -> {cache_path}")
            try:
                torch.hub.download_url_to_file(
                    self.SATLAS_AERIAL_URL,
                    cache_path,
                    progress=True,
                )
                print("[encoder] Download complete.")
            except Exception as exc:
                print(f"[encoder] WARNING: Download failed ({exc}).")
                print("[encoder] Falling back to random initialisation.")
                return
        else:
            print("[encoder] SatlasPretrain aerial weights found in cache.")

        # ── Step 2: Load and strip correct prefix ─────────────────────────
        try:
            ckpt = torch.load(cache_path, map_location="cpu", weights_only=True)

            # Unwrap common checkpoint wrappers.
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    ckpt = ckpt["model"]
                elif "state_dict" in ckpt:
                    ckpt = ckpt["state_dict"]

            # Satlas checkpoints wrap weights inside "backbone.backbone." or "backbone."
            if any(k.startswith("backbone.backbone.") for k in ckpt.keys()):
                prefix = "backbone.backbone."
            elif any(k.startswith("backbone.") for k in ckpt.keys()):
                prefix = "backbone."
            else:
                prefix = ""

            if prefix:
                stripped = {
                    k[len(prefix):]: v
                    for k, v in ckpt.items()
                    if k.startswith(prefix)
                }
            else:
                stripped = ckpt

            missing, unexpected = backbone.load_state_dict(stripped, strict=False)

            # Only the classification head should be missing — everything else
            # (all 4 Swin-v2-B transformer stages) should load cleanly.
            head_keys_missing = [k for k in missing if "head" in k]
            non_head_missing  = [k for k in missing if "head" not in k]
            if non_head_missing:
                print(f"[encoder] WARNING: {len(non_head_missing)} non-head keys missing "
                      f"from checkpoint: {non_head_missing[:5]}...")

            print(
                f"[encoder] SatlasPretrain aerial weights loaded successfully. "
                f"(skipped head keys: {len(head_keys_missing)}, "
                f"unexpected: {len(unexpected)})"
            )
        except Exception as exc:
            print(f"[encoder] WARNING: Failed to load weights ({exc}).")
            print("[encoder] Falling back to random initialisation.")

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x : (B, 3, H, W) -- ImageNet-normalised float tensor.

        Returns:
            List of 6 feature tensors ordered highest -> lowest resolution:
                [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck_s32]
        """
        # Skip 1 -- stride 1: raw input pixels (B, 3, H, W)
        skip_s1 = x

        # Skip 2 -- stride 2: lightweight CNN projection (B, 48, H/2, W/2)
        skip_s2 = self.stem_s2(x)

        # Swin-v2-B multi-scale features (tensors in B, H, W, C layout)
        swin_feats = self.swin(x)

        # Skip 3 -- stride 4:  (B, 128, H/4, W/4)
        skip_s4 = swin_feats["stage1"].permute(0, 3, 1, 2).contiguous()

        # Skip 4 -- stride 8:  (B, 256, H/8, W/8)
        skip_s8 = swin_feats["stage2"].permute(0, 3, 1, 2).contiguous()

        # Skip 5 -- stride 16: (B, 512, H/16, W/16)
        skip_s16 = swin_feats["stage3"].permute(0, 3, 1, 2).contiguous()

        # Bottleneck -- stride 32: (B, 1024, H/32, W/32)
        bottleneck = swin_feats["stage4"].permute(0, 3, 1, 2).contiguous()

        return [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck]
