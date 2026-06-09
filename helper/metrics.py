# =============================================================
#  metrics.py — Evaluation Metrics
#
#  Implements all three metrics required by the SOICT Group 24
#  project specification:
#    1. Intersection over Union (IoU / mIoU)  [primary]
#    2. Overall Pixel Accuracy                [secondary]
#    3. F1-Score / Dice Coefficient           [secondary]
#
#  Supports:
#    - Multi-class segmentation (land_cover, NUM_CLASSES=7)
#    - Binary segmentation      (road,       NUM_CLASSES=1)
#
#  Import with: from helper.metrics import SegmentationMetrics
#
#  Project: DeepGlobe Land Cover & Road Segmentation
#  Institution: SOICT, Hanoi University of Science and Technology
#  Group: 24 | Supervisor: Dr. Tran Nguyen Ngoc
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from config import CFG, LC_CLASSES


# ─────────────────────────────────────────────────────────────
# SegmentationMetrics
# ─────────────────────────────────────────────────────────────
class SegmentationMetrics:
    """
    Computes the three evaluation metrics required by the project spec
    using an accumulating confusion matrix:

        1. Mean Intersection over Union (mIoU) — PRIMARY METRIC
           IoU_c = TP_c / (TP_c + FP_c + FN_c)
           mIoU  = mean over classes with at least 1 GT pixel

        2. Overall Pixel Accuracy (OA)
           OA = sum(TP_c) / total_pixels

        3. Mean F1-Score (Dice Coefficient)
           F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)
           mF1  = mean over classes with at least 1 GT pixel

    Supports multi-class (num_classes > 1) and binary (num_classes == 1).
    For binary, a 2×2 confusion matrix is used (background vs. foreground).

    ── USAGE ──────────────────────────────────────────────────

        metric = SegmentationMetrics(num_classes=7)

        for images, masks in val_loader:
            logits = model(images)
            metric.update(logits, masks)

        results = metric.compute()   # {"miou": ..., "pixel_acc": ..., "f1": ...}
        metric.reset()               # clear for next epoch

    ────────────────────────────────────────────────────────────

    Args:
        num_classes  : Total number of segmentation classes.
                       Use 1 for binary (road) — will use a 2-class CM internally.
        ignore_index : Pixels with this label are excluded.
    """

    def __init__(self, num_classes: int = CFG["NUM_CLASSES"],
                 ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        # For binary (num_classes=1) we still use a 2×2 CM (background + foreground)
        self._cm_size = 2 if num_classes == 1 else num_classes
        # confusion[true_class, pred_class] += count
        self.confusion = np.zeros((self._cm_size, self._cm_size), dtype=np.int64)

    def reset(self):
        """Clear the confusion matrix. Call at the start of each epoch."""
        self.confusion[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate predictions into the confusion matrix.

        Args:
            preds   : (B, C, H, W) — raw logits OR softmax/sigmoid probabilities.
                      For binary (C=1): logits or sigmoid outputs.
            targets : (B, H, W)   — ground-truth integer indices (multi-class)
                      OR float binary labels in [0, 1] (binary).
        """
        if self.num_classes == 1:
            # Binary: threshold at 0.5 after sigmoid
            prob     = torch.sigmoid(preds.squeeze(1))   # (B, H, W)
            pred_cls = (prob >= 0.5).long()               # 0 or 1
            tgt_cls  = targets.long()
        else:
            pred_cls = preds.argmax(dim=1)               # (B, H, W)
            tgt_cls  = targets.long()

        pred_flat = pred_cls.cpu().numpy().ravel()
        tgt_flat  = tgt_cls.cpu().numpy().ravel()

        # Mask out ignored pixels (only applicable to multi-class)
        if self.num_classes > 1:
            valid     = tgt_flat != self.ignore_index
            pred_flat = pred_flat[valid]
            tgt_flat  = tgt_flat[valid]

        # Clamp to valid range (safety guard)
        pred_flat = np.clip(pred_flat, 0, self._cm_size - 1)
        tgt_flat  = np.clip(tgt_flat,  0, self._cm_size - 1)

        # Accumulate: confusion[true, pred] += 1
        np.add.at(self.confusion, (tgt_flat, pred_flat), 1)

    # ── Core metric computations ───────────────────────────────

    def _iou_per_class(self) -> np.ndarray:
        """IoU for each class. Returns NaN for classes with no GT pixels."""
        inter = np.diag(self.confusion)
        union = (self.confusion.sum(axis=1) +
                 self.confusion.sum(axis=0) -
                 inter)
        return np.where(union > 0, inter / union, np.nan)

    def _f1_per_class(self) -> np.ndarray:
        """F1 (Dice) for each class. Returns NaN for classes with no GT pixels."""
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / denom, np.nan)

    # ── Public API ─────────────────────────────────────────────

    def compute_miou(self) -> float:
        """Mean IoU (primary metric)."""
        return float(np.nanmean(self._iou_per_class()))

    def compute_pixel_accuracy(self) -> float:
        """Overall pixel accuracy: correct_pixels / total_pixels."""
        total   = self.confusion.sum()
        correct = np.diag(self.confusion).sum()
        if total == 0:
            return 0.0
        return float(correct / total)

    def compute_f1(self) -> float:
        """Mean F1-Score (Dice coefficient) across all classes."""
        return float(np.nanmean(self._f1_per_class()))

    def compute(self) -> dict:
        """
        Compute all three metrics at once.

        Returns:
            dict with keys:
                "miou"       : float — Mean IoU in [0, 1]
                "pixel_acc"  : float — Overall pixel accuracy in [0, 1]
                "f1"         : float — Mean F1-Score in [0, 1]
        """
        return {
            "miou":      self.compute_miou(),
            "pixel_acc": self.compute_pixel_accuracy(),
            "f1":        self.compute_f1(),
        }

    def compute_per_class(self) -> dict:
        """
        Per-class IoU and F1 scores.

        Returns:
            dict: {class_name: {"iou": ..., "f1": ...}}
                  For binary, uses "Background" and "Road"/"Foreground".
        """
        iou_arr = self._iou_per_class()
        f1_arr  = self._f1_per_class()

        if self.num_classes == 1:
            names = ["Background", "Foreground"]
        else:
            names = [LC_CLASSES[i][0] for i in range(self.num_classes)]

        return {
            name: {"iou": float(iou_arr[i]), "f1": float(f1_arr[i])}
            for i, name in enumerate(names)
        }

    def print_report(self):
        """Pretty-print the full per-class metric table and summary row."""
        per_class = self.compute_per_class()
        summary   = self.compute()

        bar_len = 20
        header  = f"  {'Class':<18}  {'IoU':>6}  {'F1':>6}  Chart"
        divider = "-" * (len(header) + bar_len)

        print("\n" + divider)
        print(header)
        print(divider)
        for name, vals in per_class.items():
            iou = vals["iou"]
            f1  = vals["f1"]
            bar = "#" * int(iou * bar_len) if not np.isnan(iou) else ""
            print(f"  {name:<18}  {iou:>6.4f}  {f1:>6.4f}  {bar}")
        print(divider)
        print(f"  {'mIoU':<18}  {summary['miou']:>6.4f}")
        print(f"  {'Pixel Acc':<18}  {summary['pixel_acc']:>6.4f}")
        print(f"  {'Mean F1':<18}  {summary['f1']:>6.4f}")
        print(divider + "\n")


# ─────────────────────────────────────────────────────────────
# Backward-compatible alias for existing code using MeanIoU
# ─────────────────────────────────────────────────────────────
class MeanIoU(SegmentationMetrics):
    """
    Backward-compatible alias for SegmentationMetrics.

    compute() now returns a dict; to get the scalar mIoU (as before),
    use compute_miou() or compute()["miou"].
    """

    def compute(self) -> float:          # type: ignore[override]
        """Returns scalar mIoU for drop-in compatibility with train.py."""
        return self.compute_miou()


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("metrics.py — Running sanity check...")

    # ── Test 1: Multi-class perfect prediction (7 classes) ────
    print("\n[Test 1] Multi-class perfect prediction:")
    C      = 7
    metric = SegmentationMetrics(num_classes=C)
    preds   = torch.eye(C).unsqueeze(0).unsqueeze(-1)      # (1, C, C, 1)
    targets = torch.arange(C).unsqueeze(0).unsqueeze(-1)   # (1, C, 1)
    metric.update(preds, targets)
    results = metric.compute()
    print(f"  mIoU      : {results['miou']:.4f}  (expected 1.0)")
    print(f"  Pixel Acc : {results['pixel_acc']:.4f}  (expected 1.0)")
    print(f"  F1        : {results['f1']:.4f}  (expected 1.0)")
    assert abs(results["miou"]      - 1.0) < 1e-4, "mIoU should be 1.0"
    assert abs(results["pixel_acc"] - 1.0) < 1e-4, "Pixel Acc should be 1.0"
    assert abs(results["f1"]        - 1.0) < 1e-4, "F1 should be 1.0"
    print("  PASSED [OK]")

    # ── Test 2: Multi-class random prediction ─────────────────
    print("\n[Test 2] Multi-class random prediction:")
    metric.reset()
    preds   = torch.randn(2, C, 64, 64)
    targets = torch.randint(0, C, (2, 64, 64))
    metric.update(preds, targets)
    results = metric.compute()
    print(f"  mIoU      : {results['miou']:.4f}  (expected ~0.14 for 7 classes)")
    print(f"  Pixel Acc : {results['pixel_acc']:.4f}")
    print(f"  F1        : {results['f1']:.4f}")
    metric.print_report()
    print("  PASSED [OK]")

    # ── Test 3: Binary (road) perfect prediction ───────────────
    print("\n[Test 3] Binary perfect prediction (road):")
    metric_bin = SegmentationMetrics(num_classes=1)
    preds   = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])  # (1, 1, 2, 2)
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])      # (1, 2, 2)
    metric_bin.update(preds, targets)
    results = metric_bin.compute()
    print(f"  mIoU      : {results['miou']:.4f}  (expected 1.0)")
    print(f"  Pixel Acc : {results['pixel_acc']:.4f}  (expected 1.0)")
    print(f"  F1        : {results['f1']:.4f}  (expected 1.0)")
    metric_bin.print_report()
    assert abs(results["miou"]      - 1.0) < 1e-4, "Binary mIoU should be 1.0"
    assert abs(results["pixel_acc"] - 1.0) < 1e-4, "Binary Pixel Acc should be 1.0"
    print("  PASSED [OK]")

    # ── Test 4: MeanIoU backward-compat alias ─────────────────
    print("\n[Test 4] MeanIoU backward-compat alias:")
    m = MeanIoU(num_classes=7)
    m.update(torch.randn(2, 7, 32, 32), torch.randint(0, 7, (2, 32, 32)))
    val = m.compute()
    assert isinstance(val, float), "MeanIoU.compute() must return float!"
    print(f"  Scalar mIoU: {val:.4f}")
    print("  PASSED [OK]")

    print("\nAll sanity checks PASSED")

