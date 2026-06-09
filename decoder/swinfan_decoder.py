# =============================================================
#  swinfan_decoder.py — SwinFAN Attention-Guided Decoder
#
#  Implements the SwinFAN decoder path:
#    AttentionGate         — spatial skip-connection refinement
#    SwinFANDecoderBlock   — upsample + attention-gated skip + conv
#    SwinFANDecoder        — full multi-scale decoder
#
#  Used by SwinFANModel in model/models.py.
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Attention Gate
# Refines skip connections before concatenation in the decoder.
# Inspired by the Attention U-Net and the SwinFAN AFRH module.
# ─────────────────────────────────────────────────────────────
class AttentionGate(nn.Module):
    """
    Spatial Attention Gate for skip connection refinement.

    At each decoder level, the gating signal (upsampled deep feature)
    is used to compute per-pixel attention weights that suppress
    irrelevant activations in the skip connection.

    This implements the attention-based feature refinement described
    in the SwinFAN paper as part of the decoder design.

    Args:
        gate_channels : Channels from the upsampled (lower-res) feature map.
        skip_channels : Channels from the encoder skip connection.
        inter_channels: Hidden channel dimension for the attention computation.
    """

    def __init__(self, gate_channels: int, skip_channels: int,
                 inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = skip_channels // 2 or 1

        # Project gating signal to inter_channels
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        # Project skip connection to inter_channels
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        # Produce a single-channel attention map [0, 1]
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate : (B, gate_channels, H, W) — upsampled deep features (gating signal).
            skip : (B, skip_channels, H, W) — encoder skip connection.

        Returns:
            Attention-weighted skip tensor (B, skip_channels, H, W).
        """
        # Align spatial sizes: gate may be slightly smaller due to integer division
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:],
                                 mode="bilinear", align_corners=False)

        g1   = self.W_gate(gate)
        x1   = self.W_skip(skip)
        psi  = self.relu(g1 + x1)
        psi  = self.psi(psi)       # (B, 1, H, W) attention map

        return skip * psi          # Weighted skip: attends to relevant features


# ─────────────────────────────────────────────────────────────
# SwinFAN Decoder Block
# Upsample × 2 + Attention-Gated skip + Conv refinement.
# ─────────────────────────────────────────────────────────────
class SwinFANDecoderBlock(nn.Module):
    """
    One SwinFAN decoder step:
        1. Upsample the incoming feature map by × 2.
        2. Refine the skip connection through an AttentionGate.
        3. Concatenate gated skip + upsampled, then apply Conv+BN+ReLU × 2.

    This is the core building block that implements the attention-guided
    multi-scale feature fusion described in the SwinFAN paper.

    Args:
        in_channels   : Channels of the upsampled bottom feature.
        skip_channels : Channels of the encoder skip connection.
        out_channels  : Channels after the convolutional refinement.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=False)
        self.attn_gate = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
        )
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
        """
        Args:
            x    : (B, in_channels, H, W) — bottom-up feature.
            skip : (B, skip_channels, H*2, W*2) — encoder skip connection.

        Returns:
            (B, out_channels, H*2, W*2) — refined feature map.
        """
        x = self.upsample(x)

        # Align spatial size if rounding errors occur
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)

        # Refine skip via attention gate using upsampled x as gating signal
        skip = self.attn_gate(gate=x, skip=skip)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────
# SwinFAN Decoder
# Progressively upsamples from bottleneck to full resolution,
# fusing attention-gated encoder skips at each level.
# ─────────────────────────────────────────────────────────────
class SwinFANDecoder(nn.Module):
    """
    SwinFAN Decoder with attention-guided skip connection fusion.

    Receives the ordered list of encoder features (highest → lowest res)
    and progressively upsamples while merging attention-gated skips.

    Args:
        encoder_channels : List of channel counts from the encoder,
                           ordered from HIGHEST to LOWEST resolution.
                           e.g. [3, 48, 96, 192, 384, 768] for SwinEncoder.
        decoder_channels : List of output channels per decoder stage.
                           e.g. (384, 192, 96, 48, 24)
    """

    def __init__(self, encoder_channels, decoder_channels=(384, 192, 96, 48, 24)):
        super().__init__()
        # Decoder starts from bottleneck (last encoder channel)
        # and fuses skips in reverse: enc[-2], enc[-3], ..., enc[0]
        in_chs   = encoder_channels[-1]
        skip_chs = list(reversed(encoder_channels[:-1]))  # e.g. [384, 192, 96, 48, 3]

        self.blocks = nn.ModuleList()
        for out_ch, skip_ch in zip(decoder_channels, skip_chs):
            self.blocks.append(SwinFANDecoderBlock(in_chs, skip_ch, out_ch))
            in_chs = out_ch

        self.out_channels = decoder_channels[-1]

    def forward(self, features):
        """
        Args:
            features : List of encoder feature maps ordered high-res to low-res.
                       e.g. [skip_s1, skip_s2, skip_s4, skip_s8, skip_s16, bottleneck]

        Returns:
            Tensor: Decoded feature map at full (stride-1) resolution.
        """
        x     = features[-1]                        # start from bottleneck
        skips = list(reversed(features[:-1]))       # [s16, s8, s4, s2, s1]

        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)

        return x
