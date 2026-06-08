import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    Lightweight Multi-Head Self-Attention block.

    Takes a CNN feature map (B, C, H, W), flattens the spatial
    dimensions into a sequence of tokens, applies Multi-Head
    Self-Attention + Feed-Forward Network, and reshapes back.

    This adds global receptive field (long-range context) to the
    otherwise locally-limited CNN features at the bottleneck.

    Usage:
        block = TransformerBlock(channels=512, num_heads=8)
        out   = block(feature_map)   # same shape as input
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn  = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # Expects (B, seq_len, embed_dim)
        )
        self.norm2 = nn.LayerNorm(channels)
        # Feed-Forward Network (expand × 4 then project back)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
        Returns:
            x: Feature map (B, C, H, W) — same shape, global context added.
        """
        B, C, H, W = x.shape

        # Step 1: Flatten (H, W) → token sequence
        tokens = x.flatten(2).transpose(1, 2)       # (B, H*W, C)

        # Step 2: Self-Attention with residual
        normed   = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens   = tokens + attn_out                 # Residual connection

        # Step 3: FFN with residual
        tokens   = tokens + self.ffn(self.norm2(tokens))

        # Step 4: Reshape back to spatial feature map
        return tokens.transpose(1, 2).reshape(B, C, H, W)
