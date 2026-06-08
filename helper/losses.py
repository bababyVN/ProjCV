# =============================================================
#  losses.py — Loss Functions (Complete, no TODOs)
#  Import with: from losses import HybridLoss
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ─────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class semantic segmentation.
    Focuses training on hard, misclassified examples by
    down-weighting well-classified pixels.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma        : Focusing parameter. Higher → more focus on
                       hard examples. Typical: 2.0
        alpha        : Base weighting factor. Typical: 0.25
        ignore_index : Pixels with this label are excluded from loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   : (B, C, H, W) — raw logits (before softmax)
            targets : (B, H, W)    — integer class indices

        Returns:
            Scalar loss tensor.
        """
        # Standard per-pixel cross-entropy (no reduction yet)
        ce_loss = F.cross_entropy(
            preds, targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )                                           # (B, H, W)

        # p_t = probability assigned to the correct class
        pt = torch.exp(-ce_loss)

        # Focal weight: down-weights easy pixels (high pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Directly optimises the Dice coefficient (= F1 score per class)
    and handles class imbalance better than cross-entropy alone.

    Dice = (2 * |X ∩ Y|) / (|X| + |Y|)
    Loss = 1 - mean(Dice per class)

    Args:
        num_classes  : Number of segmentation classes.
        smooth       : Small constant to prevent division by zero.
        ignore_index : Pixels with this label are excluded.
    """

    def __init__(self, num_classes: int = 7,
                 smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   : (B, C, H, W) — raw logits
            targets : (B, H, W)    — integer class indices

        Returns:
            Scalar loss tensor.
        """
        # Softmax → per-class probabilities
        probs = F.softmax(preds, dim=1)             # (B, C, H, W)

        # Build valid-pixel mask (exclude ignore_index)
        valid   = (targets != self.ignore_index)    # (B, H, W) bool
        targets = targets.clone()
        targets[~valid] = 0                         # neutral class (won't affect dice)

        # One-hot encode targets: (B, H, W) → (B, H, W, C) → (B, C, H, W)
        one_hot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Zero out ignored pixels in both tensors
        valid_4d = valid.unsqueeze(1).float()           # (B, 1, H, W)
        probs    = probs    * valid_4d
        one_hot  = one_hot  * valid_4d

        # Per-class Dice (sum over B, H, W dims → one value per class)
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))   # (C,)
        cardinality  = (probs + one_hot).sum(dim=(0, 2, 3))   # (C,)

        dice_per_class = (2 * intersection + self.smooth) / \
                         (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


# ─────────────────────────────────────────────────────────────
# Hybrid Loss  (Focal + Dice)
# ─────────────────────────────────────────────────────────────
class HybridLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Dice Loss.

    Using both losses together is a common best-practice in
    medical and satellite image segmentation:
        • Focal Loss → handles per-pixel class imbalance
        • Dice Loss  → directly optimises segmentation overlap metric

    Default weights (focal=0.5, dice=0.5) are a good starting point.
    You can tune these in CFG["FOCAL_WEIGHT"] / CFG["DICE_WEIGHT"].
    """

    def __init__(self, num_classes: int = CFG["NUM_CLASSES"],
                 focal_weight: float = CFG["FOCAL_WEIGHT"],
                 dice_weight:  float = CFG["DICE_WEIGHT"],
                 focal_gamma:  float = CFG["FOCAL_GAMMA"],
                 focal_alpha:  float = CFG["FOCAL_ALPHA"],
                 ignore_index: int   = 255):
        super().__init__()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha,
                               ignore_index=ignore_index)
        self.dice  = DiceLoss(num_classes=num_classes,
                              ignore_index=ignore_index)
        self.fw = focal_weight
        self.dw = dice_weight

    def forward(self, preds: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds   : (B, C, H, W) — raw logits
            targets : (B, H, W)    — integer class indices

        Returns:
            Scalar combined loss.
        """
        focal = self.focal(preds, targets)
        dice  = self.dice(preds, targets)
        return self.fw * focal + self.dw * dice


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("losses.py — Running sanity check...")
    B, C, H, W = 2, 7, 512, 512
    preds   = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    criterion = HybridLoss(num_classes=C)
    loss      = criterion(preds, targets)
    print(f"HybridLoss output: {loss.item():.4f}  (scalar)")
    assert loss.ndim == 0, "Loss must be a scalar!"
    print("Sanity check PASSED")
