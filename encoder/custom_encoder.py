import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from config import CFG
from encoder.transformer_block import TransformerBlock


class CustomHybridEncoder(nn.Module):
    """
    Hybrid Encoder combining a pretrained ResNet-34 CNN backbone
    with a Transformer self-attention bottleneck.

    Architecture
    ------------
    Input: (B, 3, H, W)

    Stage  | Module          | Output shape      | Stride
    -------|-----------------|-------------------|-------
    stem   | Conv3x3 BN ReLU | (B,  64, H,    W) |  x1
    s2     | ResNet conv1    | (B,  64, H/2,  W) |  x2
    s4     | maxpool+layer1  | (B,  64, H/4,  W) |  x4
    s8     | layer2          | (B, 128, H/8,  W) |  x8
    s16    | layer3          | (B, 256, H/16, W) | x16
    bot    | TransformerBlock| (B, 256, H/16, W) | x16

    Returns: [skip1, skip2, skip3, skip4, bottleneck]
    encoder_channels = [64, 64, 64, 128, 256]

    Design Rationale
    ----------------
    - ResNet-34 is chosen over ResNet-50 for lower VRAM use on GPU T4/P100.
    - layer4 (512 ch, stride 32) is deliberately omitted: on 512x512 inputs
      it would yield 16x16=256 tokens only, losing too much spatial detail
      for satellite segmentation.
    - A lightweight 3x3 stem (stride 1) provides the highest-resolution
      skip that standard ResNet architectures lack.
    - The TransformerBlock at stride 16 operates on 32x32=1024 tokens,
      giving global receptive field at manageable quadratic cost.
    """

    def __init__(self, in_channels: int = 3, config: dict = CFG):
        super().__init__()
        self.config = config

        # ------------------------------------------------------------------
        # Load ResNet-34 backbone (ImageNet pretrained, fallback to random)
        # ------------------------------------------------------------------
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            print("[encoder] ResNet-34: ImageNet pretrained weights loaded.")
        except Exception:
            from torchvision.models import resnet34
            resnet = resnet34(weights=None)
            print("[encoder] WARNING: pretrained weights unavailable -- "
                  "using random initialisation.")

        # ------------------------------------------------------------------
        # Stage 0 -- stem: stride-1 full-resolution skip (3 -> 64 ch)
        # Provides the highest-resolution skip connection that ResNet lacks.
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # Stage 1 -- encoder_s2: ResNet 7x7 conv, stride 2  (B, 64, H/2, W/2)
        # ------------------------------------------------------------------
        self.encoder_s2 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        # ------------------------------------------------------------------
        # Stage 2 -- encoder_s4: maxpool + layer1, stride 4 (B, 64, H/4, W/4)
        # layer1 = 3x BasicBlock(64->64), no internal downsampling.
        # ------------------------------------------------------------------
        self.encoder_s4 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1,
        )

        # ------------------------------------------------------------------
        # Stage 3 -- encoder_s8: layer2, stride 8           (B, 128, H/8, W/8)
        # layer2 = 4x BasicBlock(64->128), first block stride-2.
        # ------------------------------------------------------------------
        self.encoder_s8 = resnet.layer2

        # ------------------------------------------------------------------
        # Stage 4 -- encoder_s16: layer3, stride 16         (B, 256, H/16, W/16)
        # layer3 = 6x BasicBlock(128->256), first block stride-2.
        # ------------------------------------------------------------------
        self.encoder_s16 = resnet.layer3

        # ------------------------------------------------------------------
        # Transformer Bottleneck -- global self-attention at stride 16
        # On IMG_SIZE=512: 512/16 = 32 -> 32x32 = 1024 tokens.
        # channels=256 must match encoder_s16 output channels.
        # ------------------------------------------------------------------
        self.transformer = TransformerBlock(channels=256, num_heads=8,
                                            dropout=0.1)

        # ------------------------------------------------------------------
        # Channel spec consumed by UNetDecoder to build its blocks.
        # Order: [stem, s2, s4, s8, s16_bottleneck]
        # ------------------------------------------------------------------
        self.encoder_channels = [64, 64, 64, 128, 256]

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: (B, 3, H, W) ImageNet-normalised float tensor.

        Returns:
            List of 5 feature tensors ordered highest -> lowest resolution:
                [skip1, skip2, skip3, skip4, bottleneck]
        """
        # Step 1: full-resolution skip  (B, 64, H, W)
        skip1 = self.stem(x)

        # Step 2: stride-2 skip         (B, 64, H/2, W/2)
        skip2 = self.encoder_s2(x)

        # Step 3: stride-4 skip         (B, 64, H/4, W/4)
        skip3 = self.encoder_s4(skip2)

        # Step 4: stride-8 skip         (B, 128, H/8, W/8)
        skip4 = self.encoder_s8(skip3)

        # Step 5: stride-16 CNN features (B, 256, H/16, W/16)
        bottleneck = self.encoder_s16(skip4)

        # Step 6: enrich with global self-attention context
        bottleneck = self.transformer(bottleneck)

        return [skip1, skip2, skip3, skip4, bottleneck]
