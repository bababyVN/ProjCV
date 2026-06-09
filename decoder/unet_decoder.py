# =============================================================
#  unet_decoder.py — U-Net Style Decoder
#
#  Standard U-Net Decoder with skip connections (retained for legacy
#  "hybrid" architecture).
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Standard U-Net Decoder with skip connections (legacy, no attention)."""

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32)):
        super().__init__()
        in_chs   = encoder_channels[-1]
        skip_chs = list(reversed(encoder_channels[:-1]))

        self.blocks = nn.ModuleList()
        for out_ch, skip_ch in zip(decoder_channels, skip_chs):
            self.blocks.append(UNetDecoderBlock(in_chs, skip_ch, out_ch))
            in_chs = out_ch

        self.out_channels = decoder_channels[-1]

    def forward(self, features):
        x     = features[-1]
        skips = list(reversed(features[:-1]))
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return x
