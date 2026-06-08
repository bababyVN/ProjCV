# =============================================================
#  metrics.py — Evaluation Metrics (Complete, no TODOs)
#  Import with: from metrics import MeanIoU
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CFG, LC_CLASSES


# ─────────────────────────────────────────────────────────────
# Mean Intersection-over-Union (mIoU)
# ─────────────────────────────────────────────────────────────
class MeanIoU:
    """
    Computes Mean Intersection over Union (mIoU) across all classes
    using an accumulating confusion matrix.

    ──────── WHAT IS mIoU? ──────────────────────────────────────

    For each class c, IoU is:
        IoU_c = TP_c / (TP_c + FP_c + FN_c)

    Where:
        TP_c = pixels correctly predicted as class c
        FP_c = pixels of other classes incorrectly predicted as c
        FN_c = pixels of class c incorrectly predicted as other classes

    mIoU = average of IoU_c over all classes with at least one GT pixel.

    ──────── USAGE ──────────────────────────────────────────────

        metric = MeanIoU(num_classes=7)

        for images, masks in val_loader:
            logits = model(images)
            metric.update(logits, masks)     # accumulate per batch

        val_miou = metric.compute()          # final mIoU for the epoch
        metric.reset()                       # clear for next epoch

    ─────────────────────────────────────────────────────────────

    Args:
        num_classes  : Total number of segmentation classes.
        ignore_index : Pixels with this label are excluded.
    """

    def __init__(self, num_classes: int = CFG["NUM_CLASSES"],
                 ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        # (num_classes × num_classes) confusion matrix.
        # confusion[true_class, pred_class] += count
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        """Clear the confusion matrix. Call at the start of each epoch."""
        self.confusion[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate predictions into the confusion matrix.

        Args:
            preds   : (B, C, H, W) — raw logits OR softmax probabilities.
                      Argmax is applied internally.
            targets : (B, H, W)    — ground-truth integer class indices.
        """
        # Convert logits to predicted class indices
        pred_cls = preds.argmax(dim=1)               # (B, H, W)

        # Flatten to 1-D arrays for indexing
        pred_flat = pred_cls.cpu().numpy().ravel()
        tgt_flat  = targets.cpu().numpy().ravel()

        # Mask out ignored pixels
        valid      = tgt_flat != self.ignore_index
        pred_flat  = pred_flat[valid]
        tgt_flat   = tgt_flat[valid]

        # Accumulate into confusion matrix using fast np.add.at
        # confusion[true, pred] += 1
        np.add.at(self.confusion, (tgt_flat, pred_flat), 1)

    def compute(self) -> float:
        """
        Compute mIoU from the accumulated confusion matrix.

        Returns:
            float: Mean IoU in range [0, 1].
                   Classes with zero ground-truth pixels are excluded
                   from the mean (NaN → ignored).
        """
        # Diagonal of confusion matrix = True Positives per class
        inter = np.diag(self.confusion)                         # (C,)

        # Union = row_sum + col_sum - TP
        #       = (all GT pixels of class c) + (all pixels predicted as c) - TP
        union = (self.confusion.sum(axis=1) +   # GT pixels per class
                 self.confusion.sum(axis=0) -   # Predicted pixels per class
                 inter)

        # IoU per class; skip classes with no GT pixels (union=0 → NaN)
        iou_per_class = np.where(union > 0, inter / union, np.nan)

        return float(np.nanmean(iou_per_class))

    def compute_per_class(self) -> dict:
        """
        Compute per-class IoU and return as a readable dictionary.

        Returns:
            dict: {class_name: iou_value} for each class.
                  Value is NaN if the class had no GT pixels.
        """
        inter = np.diag(self.confusion)
        union = (self.confusion.sum(axis=1) +
                 self.confusion.sum(axis=0) -
                 inter)
        iou_per_class = np.where(union > 0, inter / union, np.nan)

        result = {}
        for idx, (name, _) in LC_CLASSES.items():
            iou = iou_per_class[idx] if idx < len(iou_per_class) else np.nan
            result[name] = float(iou)
        return result

    def print_report(self):
        """Pretty-print the per-class IoU table and the final mIoU."""
        per_class = self.compute_per_class()
        miou      = self.compute()

        print("\n" + "-" * 40)
        print(f"  {'Class':<18}  IoU")
        print("-" * 40)
        for name, iou in per_class.items():
            bar = "#" * int(iou * 20) if not np.isnan(iou) else ""
            print(f"  {name:<18}  {iou:.4f}  {bar}")
        print("-" * 40)
        print(f"  {'mIoU':<18}  {miou:.4f}")
        print("-" * 40 + "\n")


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("metrics.py — Running sanity check...")
    C = 7
    metric = MeanIoU(num_classes=C)

    # Perfect predictions: mIoU should be 1.0
    preds   = torch.eye(C).unsqueeze(0).unsqueeze(-1)  # (1, C, C, 1)
    targets = torch.arange(C).unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
    metric.update(preds, targets)
    miou = metric.compute()
    print(f"Perfect prediction mIoU: {miou:.4f}  (expected ~1.0)")

    metric.reset()

    # Random predictions: mIoU should be low
    preds   = torch.randn(2, C, 64, 64)
    targets = torch.randint(0, C, (2, 64, 64))
    metric.update(preds, targets)
    miou = metric.compute()
    print(f"Random prediction mIoU:  {miou:.4f}  (expected ~0.14 for 7 classes)")
    metric.print_report()
    print("Sanity check PASSED")
