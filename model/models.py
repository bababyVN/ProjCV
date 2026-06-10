# =============================================================
#  models.py — SwinFAN Model Architecture
#
#  Implements the SwinFAN (Swin-based Focal Axial attention Network)
#  hybrid encoder-decoder model for semantic segmentation.
#
#  Architecture:
#    SwinEncoder → SwinFANDecoder (with AttentionGate at each level)
#                → SegmentationHead → Logits
#
#  Encoders live in:  encoder/
#  Decoders live in:  decoder/   (swinfan_decoder.py, unet_decoder.py)
#
#  Also retains the legacy HybridSegModel (ResNet-34 + TransformerBlock)
#  for backward compatibility ("hybrid" arch setting in CFG).
#
#  Import with: from model.models import build_model
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from encoder.custom_encoder  import CustomHybridEncoder
from encoder.swin_encoder    import SwinEncoder, SwinBaseEncoder, SwinV2BaseEncoder
from decoder.swinfan_decoder import SwinFANDecoder
from decoder.unet_decoder    import UNetDecoder


# ─────────────────────────────────────────────────────────────
# Decoder classes are defined in decoder/
# (SwinFANDecoder, UNetDecoder — imported at top of file)
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# SHARED: Segmentation Head
# ─────────────────────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """
    Final prediction head.

    For multi-class (land_cover):
        1×1 Conv maps features → num_classes logits.
        Upsample to input resolution.

    For binary (road):
        num_classes=1, outputs a single channel logit map.
        Apply sigmoid externally for probabilities.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor,
                target_size: tuple = (512, 512)) -> torch.Tensor:
        logits = self.conv(x)
        return F.interpolate(logits, size=target_size,
                             mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────
# SwinFAN Model — Full End-to-End Model
# ─────────────────────────────────────────────────────────────
class SwinFANModel(nn.Module):
    """
    SwinFAN: Swin-based Focal Axial attention Network for semantic segmentation.

    Pipeline:
        Input → SwinEncoder → SwinFANDecoder → SegmentationHead → Logits

    Supports:
        - Multi-class segmentation (land_cover, num_classes=7)
        - Binary segmentation     (road,       num_classes=1)

    Args:
        encoder        : SwinEncoder instance.
        num_classes    : Number of output classes (7 or 1).
        decoder_channels : Channel progression of the decoder.
    """

    def __init__(self, encoder: nn.Module, num_classes: int,
                 decoder_channels=(384, 192, 96, 48, 24)):
        super().__init__()
        self.encoder = encoder
        self.decoder = SwinFANDecoder(
            encoder_channels=encoder.encoder_channels,
            decoder_channels=decoder_channels,
        )
        self.head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            num_classes=num_classes,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input batch (B, 3, H, W)

        Returns:
            logits : (B, num_classes, H, W) — raw unnormalised scores.
                     • Multi-class: apply softmax / argmax for inference.
                     • Binary:      apply sigmoid for probability.
        """
        target_size = x.shape[2:]           # (H, W) of the input patch
        features    = self.encoder(x)       # list of 6 multi-scale tensors
        decoded     = self.decoder(features)
        logits      = self.head(decoded, target_size)
        return logits


class HybridSegModel(nn.Module):
    """Legacy model: CustomHybridEncoder → UNetDecoder → SegmentationHead."""

    def __init__(self, encoder: nn.Module, num_classes: int,
                 decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder(
            encoder_channels=encoder.encoder_channels,
            decoder_channels=decoder_channels,
        )
        self.head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[2:]
        features    = self.encoder(x)
        decoded     = self.decoder(features)
        logits      = self.head(decoded, target_size)
        return logits


# ─────────────────────────────────────────────────────────────
# Factory function — called from train.py
# ─────────────────────────────────────────────────────────────
def build_model(config: dict = CFG) -> nn.Module:
    """
    Assemble and return the full segmentation model.

    Called in train.py:
        model = build_model(CFG).to(device)

    Reads CFG["ARCH"] to select the architecture:
        "swinfan"    → SwinFANModel  (Swin-T encoder + attention-guided decoder)
        "swinfan_v2" → SwinFANModel  (Swin-B encoder, ImageNet-1K pretrain)
        "swinfan_v3" → SwinFANModel  (Swin-v2-B encoder, SatlasPretrain aerial)
        "hybrid"     → HybridSegModel (ResNet-34 CNN encoder + standard UNetDecoder)

    Args:
        config : CFG dict from config.py.

    Returns:
        nn.Module: Ready-to-train segmentation model.
    """
    arch        = config.get("ARCH", "swinfan").lower()
    num_classes = config["NUM_CLASSES"]

    if arch == "swinfan":
        # ── SwinFAN-v1: Swin-T (28M params) + Attention-Guided Decoder ──────
        encoder = SwinEncoder(in_channels=3, config=config)
        model   = SwinFANModel(
            encoder=encoder,
            num_classes=num_classes,
            decoder_channels=(384, 192, 96, 48, 24),
        )
        label = "SwinFANModel (v1 — Swin-T)"

    elif arch == "swinfan_v2":
        # ── SwinFAN-v2: Swin-B (88M params) + Wider Attention-Guided Decoder ─
        # Swin-B encoder channels: [3, 48, 128, 256, 512, 1024]
        # Decoder channels scaled up to match wider encoder skips.
        encoder = SwinBaseEncoder(in_channels=3, config=config)
        model   = SwinFANModel(
            encoder=encoder,
            num_classes=num_classes,
            decoder_channels=(512, 256, 128, 64, 32),
        )
        label = "SwinFANModel (v2 — Swin-B, ImageNet-1K)"

    elif arch == "swinfan_v3":
        # ── SwinFAN-v3: Swin-v2-B + SatlasPretrain Aerial Weights ────────────
        # Uses SwinV2BaseEncoder which downloads aerial_swinb_si.pth (~503 MB)
        # from HuggingFace on first run and caches it locally.
        # Encoder channels: [3, 48, 128, 256, 512, 1024] — identical to v2.
        # Decoder channels unchanged from v2.
        encoder = SwinV2BaseEncoder(in_channels=3, config=config)
        model   = SwinFANModel(
            encoder=encoder,
            num_classes=num_classes,
            decoder_channels=(512, 256, 128, 64, 32),
        )
        label = "SwinFANModel (v3 — Swin-v2-B, SatlasPretrain Aerial)"

    else:
        # ── Legacy Hybrid: ResNet-34 CNN + Transformer + UNetDecoder ─────────
        encoder = CustomHybridEncoder(in_channels=3, config=config)
        model   = HybridSegModel(
            encoder=encoder,
            num_classes=num_classes,
            decoder_channels=(256, 128, 64, 32),
        )
        label = "HybridSegModel (legacy)"

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[models] {label} | Task: {config.get('TASK')} "
          f"| Classes: {num_classes} | Trainable params: {total:,}")
    return model


# ─────────────────────────────────────────────────────────────
# Quick test — run `python model/models.py`
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import copy

    print("=" * 60)
    print("  models.py — SwinFAN Architecture Shape Test")
    print("=" * 60)

    dummy = torch.randn(2, 3, 512, 512)

    # ── Test 1: SwinFAN-v1 — Land Cover (7 classes) ───────────
    print("\n[Test 1] SwinFAN-v1 (Swin-T) — land_cover (7 classes)")
    cfg_lc = copy.deepcopy(CFG)
    cfg_lc["ARCH"]        = "swinfan"
    cfg_lc["TASK"]        = "land_cover"
    cfg_lc["NUM_CLASSES"] = 7
    model  = build_model(cfg_lc)
    output = model(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (2, 7, 512, 512), f"Shape mismatch! Got {output.shape}"
    print("  PASSED [OK]")

    # ── Test 2: SwinFAN-v1 — Road (binary, 1 class) ───────────
    print("\n[Test 2] SwinFAN-v1 (Swin-T) — road (1 class binary)")
    cfg_road = copy.deepcopy(CFG)
    cfg_road["ARCH"]        = "swinfan"
    cfg_road["TASK"]        = "road"
    cfg_road["NUM_CLASSES"] = 1
    model  = build_model(cfg_road)
    output = model(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (2, 1, 512, 512), f"Shape mismatch! Got {output.shape}"
    print("  PASSED [OK]")

    # ── Test 3: SwinFAN-v2 — Land Cover (7 classes) ───────────
    print("\n[Test 3] SwinFAN-v2 (Swin-B) — land_cover (7 classes)")
    cfg_v2 = copy.deepcopy(CFG)
    cfg_v2["ARCH"]        = "swinfan_v2"
    cfg_v2["TASK"]        = "land_cover"
    cfg_v2["NUM_CLASSES"] = 7
    model  = build_model(cfg_v2)
    output = model(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (2, 7, 512, 512), f"Shape mismatch! Got {output.shape}"
    print("  PASSED [OK]")

    # ── Test 4: SwinFAN-v3 shape test (no download — random init) ────────────
    # We test v3 with a monkeypatched _load_satlas_weights that does nothing,
    # so no 503 MB download is triggered during local shape testing.
    print("\n[Test 4] SwinFAN-v3 (Swin-v2-B, no download) — land_cover (7 classes)")
    from encoder.swin_encoder import SwinV2BaseEncoder as _V3Enc
    _orig_load = _V3Enc._load_satlas_weights
    _V3Enc._load_satlas_weights = lambda self, bb: None   # skip download
    cfg_v3 = copy.deepcopy(CFG)
    cfg_v3["ARCH"]        = "swinfan_v3"
    cfg_v3["TASK"]        = "land_cover"
    cfg_v3["NUM_CLASSES"] = 7
    model  = build_model(cfg_v3)
    output = model(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (2, 7, 512, 512), f"Shape mismatch! Got {output.shape}"
    print("  PASSED [OK]")
    _V3Enc._load_satlas_weights = _orig_load   # restore

    # ── Test 5: Legacy Hybrid model ───────────────────────────
    print("\n[Test 5] Legacy HybridSegModel — land_cover (7 classes)")
    cfg_hyb = copy.deepcopy(CFG)
    cfg_hyb["ARCH"]        = "hybrid"
    cfg_hyb["TASK"]        = "land_cover"
    cfg_hyb["NUM_CLASSES"] = 7
    model  = build_model(cfg_hyb)
    output = model(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(output.shape)}")
    assert output.shape == (2, 7, 512, 512), f"Shape mismatch! Got {output.shape}"
    print("  PASSED [OK]")

    print("\n" + "=" * 60)
    print("  All shape tests PASSED")
    print("=" * 60)

