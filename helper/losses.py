# =============================================================
#  losses.py — Loss Functions (Multi-class & Binary)
#
#  Supports both:
#    - Multi-class segmentation  (land_cover, NUM_CLASSES=7)
#    - Binary segmentation       (road,       NUM_CLASSES=1)
#
#  HybridLoss = FocalLoss + DiceLoss (weighted combination)
#
#  Import with: from helper.losses import HybridLoss
#
#  Project: DeepGlobe Land Cover & Road Segmentation
#  Institution: SOICT, Hanoi University of Science and Technology
#  Group: 24 | Supervisor: Dr. Tran Nguyen Ngoc
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


# ─────────────────────────────────────────────────────────────
# Focal Loss (multi-class AND binary)
# ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss for semantic segmentation.

    Focuses training on hard, misclassified examples by
    down-weighting well-classified pixels.

        FL(p_t) = −α · (1 − p_t)^γ · log(p_t)

    Supports:
        • Multi-class : preds shape (B, C, H, W), C > 1
                        Uses F.cross_entropy internally.
        • Binary      : preds shape (B, 1, H, W) or (B, H, W), C = 1
                        Uses F.binary_cross_entropy_with_logits.

    Args:
        gamma        : Focusing parameter. Higher → more focus on hard
                       examples. Typical value: 2.0
        alpha        : Base weighting factor. Typical value: 0.25
        ignore_index : Pixels with this label are excluded (multi-class only).
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
            preds   : (B, C, H, W) — raw logits (before softmax/sigmoid).
            targets : (B, H, W)    — integer class indices (multi-class)
                      OR float binary labels in [0, 1] (binary).

        Returns:
            Scalar focal loss tensor.
        """
        num_classes = preds.shape[1]

        if num_classes == 1:
            # ── Binary mode (road extraction) ─────────────────
            # Squeeze to (B, H, W), targets must be float [0, 1]
            preds_sq = preds.squeeze(1)                             # (B, H, W)
            targets_f = targets.float()

            bce = F.binary_cross_entropy_with_logits(
                preds_sq, targets_f, reduction="none"
            )                                                       # (B, H, W)
            pt          = torch.exp(-bce)
            focal_loss  = self.alpha * (1 - pt) ** self.gamma * bce
            return focal_loss.mean()

        else:
            # ── Multi-class mode (land_cover) ─────────────────
            ce_loss = F.cross_entropy(
                preds, targets.long(),
                ignore_index=self.ignore_index,
                reduction="none",
            )                                                       # (B, H, W)
            pt          = torch.exp(-ce_loss)
            focal_loss  = self.alpha * (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()


# ─────────────────────────────────────────────────────────────
# Dice Loss (multi-class AND binary)
# ─────────────────────────────────────────────────────────────
class DiceLoss(nn.Module):
    """
    Soft Dice Loss for semantic segmentation.

    Directly optimises the Dice coefficient (= F1 score per class)
    which handles class imbalance better than cross-entropy alone.

        Dice   = (2 · |X ∩ Y|) / (|X| + |Y|)
        Loss   = 1 − mean(Dice per class)

    Supports:
        • Multi-class : preds (B, C, H, W), C > 1 → one-hot + softmax
        • Binary      : preds (B, 1, H, W), C = 1 → sigmoid

    Args:
        num_classes  : Number of segmentation classes.
        smooth       : Small constant to prevent division by zero.
        ignore_index : Pixels with this label are excluded (multi-class only).
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
            preds   : (B, C, H, W) — raw logits.
            targets : (B, H, W)    — integer class indices (multi-class)
                      OR float binary labels in [0, 1] (binary).

        Returns:
            Scalar Dice loss tensor.
        """
        if self.num_classes == 1:
            # ── Binary mode ───────────────────────────────────
            probs    = torch.sigmoid(preds.squeeze(1))   # (B, H, W)
            targets_f = targets.float()

            inter = (probs * targets_f).sum(dim=(1, 2))  # (B,)
            card  = (probs + targets_f).sum(dim=(1, 2))  # (B,)
            dice  = (2 * inter + self.smooth) / (card + self.smooth)
            return 1.0 - dice.mean()

        else:
            # ── Multi-class mode ──────────────────────────────
            probs = F.softmax(preds, dim=1)              # (B, C, H, W)

            # Build valid-pixel mask
            valid   = (targets != self.ignore_index)     # (B, H, W) bool
            targets_c = targets.clone().long()
            targets_c[~valid] = 0

            # One-hot encode: (B, H, W) → (B, C, H, W)
            one_hot = F.one_hot(targets_c, self.num_classes)
            one_hot = one_hot.permute(0, 3, 1, 2).float()

            # Zero out ignored pixels
            valid_4d = valid.unsqueeze(1).float()
            probs    = probs    * valid_4d
            one_hot  = one_hot  * valid_4d

            # Per-class Dice
            inter = (probs * one_hot).sum(dim=(0, 2, 3))  # (C,)
            card  = (probs + one_hot).sum(dim=(0, 2, 3))  # (C,)
            dice  = (2 * inter + self.smooth) / (card + self.smooth)
            return 1.0 - dice.mean()


# ─────────────────────────────────────────────────────────────
# Hybrid Loss (Focal + Dice) — primary training criterion
# ─────────────────────────────────────────────────────────────
class HybridLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Dice Loss.

    Both losses together are a best-practice for satellite segmentation:
        • Focal Loss → handles per-pixel class imbalance
        • Dice Loss  → directly optimises overlap (IoU-related metric)

    Automatically adapts to multi-class or binary segmentation based
    on num_classes (multi-class if > 1, binary if == 1).

    Default weights (focal=0.5, dice=0.5) are a good starting point;
    adjust via CFG["FOCAL_WEIGHT"] / CFG["DICE_WEIGHT"].
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
            preds   : (B, C, H, W) — raw logits.
            targets : (B, H, W)    — ground-truth labels.

        Returns:
            Scalar combined loss: focal_weight * Focal + dice_weight * Dice.
        """
        focal = self.focal(preds, targets)
        dice  = self.dice(preds, targets)
        return self.fw * focal + self.dw * dice


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("losses.py — Running sanity check...")

    # ── Multi-class (land_cover, 7 classes) ───────────────────
    print("\n[Test 1] Multi-class HybridLoss (7 classes):")
    B, C, H, W = 2, 7, 512, 512
    preds   = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))
    loss    = HybridLoss(num_classes=C)(preds, targets)
    print(f"  HybridLoss output: {loss.item():.4f}  (scalar)")
    assert loss.ndim == 0, "Loss must be a scalar!"
    print("  PASSED [OK]")

    # ── Binary (road, 1 class) ─────────────────────────────────
    print("\n[Test 2] Binary HybridLoss (1 class, road):")
    B, H, W = 2, 512, 512
    preds   = torch.randn(B, 1, H, W)
    targets = torch.randint(0, 2, (B, H, W)).float()
    loss    = HybridLoss(num_classes=1)(preds, targets)
    print(f"  HybridLoss output: {loss.item():.4f}  (scalar)")
    assert loss.ndim == 0, "Loss must be a scalar!"
    print("  PASSED [OK]")

    print("\nAll sanity checks PASSED")

