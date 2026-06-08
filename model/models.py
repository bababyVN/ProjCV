# =============================================================
#  models.py — Model Architecture (Hybrid CNN + Transformer)
#
#  ★ YOU IMPLEMENT: CustomHybridEncoder.forward()
#  ✓ PROVIDED:      UNetDecoder, SegmentationHead, HybridSegModel,
#                   build_model()
#
#  Import with: from models import build_model
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG
from encoder.custom_encoder import CustomHybridEncoder


# ─────────────────────────────────────────────────────────────
# PROVIDED: U-Net Style Decoder
# Takes skip connections from the encoder and upsamples them.
# ─────────────────────────────────────────────────────────────
class UNetDecoderBlock(nn.Module):
    """Single upsampling step: upsample × 2 + concat skip + Conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Handle odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    Standard U-Net Decoder with skip connections.

    Receives the list of encoder features (multi-scale) and
    progressively upsamples while merging skip connections until
    the original image resolution is restored.

    Args:
        encoder_channels : List of channel counts from the encoder,
                           ordered from HIGHEST to LOWEST resolution.
                           e.g. [64, 64, 128, 256, 512] for ResNet-34.
        decoder_channels : List of output channels per decoder stage.
                           e.g. [256, 128, 64, 32]
    """

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        # encoder_channels = [c0(high-res), c1, c2, c3, c4(bottleneck)]
        # Decoder starts from bottleneck (c4) and fuses skip connections
        # in reverse order: c3, c2, c1, c0.
        in_chs   = encoder_channels[-1]     # bottleneck channels
        skip_chs = list(reversed(encoder_channels[:-1]))  # [c3, c2, c1, c0]

        self.blocks = nn.ModuleList()
        for out_ch, skip_ch in zip(decoder_channels, skip_chs):
            self.blocks.append(UNetDecoderBlock(in_chs, skip_ch, out_ch))
            in_chs = out_ch

        self.out_channels = decoder_channels[-1]

    def forward(self, features):
        """
        Args:
            features: List of encoder feature maps, ordered
                      from high-res to low-res (bottleneck last).
                      e.g. [f0(512), f1(256), f2(128), f3(64), bottleneck(64)]
        Returns:
            Tensor: Decoded feature map at ~original resolution.
        """
        # Start from the bottleneck
        x     = features[-1]
        skips = list(reversed(features[:-1]))   # [f3, f2, f1, f0]

        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)

        return x


# ─────────────────────────────────────────────────────────────
# PROVIDED: Final Segmentation Head
# ─────────────────────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """
    1×1 convolution to map decoder features → class logits.
    Then upsampled to match the input image size.
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
# PROVIDED: Full Segmentation Model
# Wires Encoder → Decoder → Head together.
# ─────────────────────────────────────────────────────────────
class HybridSegModel(nn.Module):
    """
    End-to-end segmentation model:
        Input  → CustomHybridEncoder → UNetDecoder → SegmentationHead → Logits

    You only need to implement CustomHybridEncoder.forward().
    Everything else is wired up for you here.
    """

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
        """
        Args:
            x: Input batch (B, 3, H, W)
        Returns:
            logits: (B, NUM_CLASSES, H, W) — raw unnormalised scores.
                    Apply softmax (or argmax for inference).
        """
        target_size = x.shape[2:]           # (H, W) of input patch
        features    = self.encoder(x)       # list of multi-scale tensors
        decoded     = self.decoder(features)
        logits      = self.head(decoded, target_size)
        return logits


# ─────────────────────────────────────────────────────────────
# PROVIDED: Factory function called from train.py
# ─────────────────────────────────────────────────────────────
def build_model(config: dict = CFG) -> nn.Module:
    """
    Assemble and return the full segmentation model.

    Called in train.py:
        model = build_model(CFG).to(device)

    Args:
        config: CFG from config.py.

    Returns:
        HybridSegModel: Ready-to-train model.
    """
    encoder = CustomHybridEncoder(in_channels=3, config=config)

    model = HybridSegModel(
        encoder=encoder,
        num_classes=config["NUM_CLASSES"],
        decoder_channels=(256, 128, 64, 32),
    )

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[models] HybridSegModel | Trainable params: {total:,}")
    return model


# ─────────────────────────────────────────────────────────────
# Quick test — run `python models.py` after implementing encoder
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("models.py — Running shape test...")
    model  = build_model(CFG)
    dummy  = torch.randn(2, 3, 512, 512)
    output = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == (2, CFG["NUM_CLASSES"], 512, 512), \
        f"Shape mismatch! Expected (2, {CFG['NUM_CLASSES']}, 512, 512)"
    print("Shape test PASSED")
