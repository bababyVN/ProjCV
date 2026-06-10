# =============================================================
#  evaluate_model.py — Detailed Per-Class Evaluation Script
#
#  Loads a trained model checkpoint (.pth) and calculates:
#    1. Overall Pixel Accuracy
#    2. 7-Class mIoU & Mean F1 (with "Unknown")
#    3. 6-Class mIoU & Mean F1 (excluding "Unknown")
#    4. Detailed per-class IoU & F1 table with visual chart
#
#  Run from project root:
#      python scripts/evaluate_model.py [path_to_model.pth]
# =============================================================

import sys
from pathlib import Path
import os

# Set UTF-8 encoding on Windows console for table/chart printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from tqdm import tqdm

from config import CFG, LC_CLASSES
from dataloader.dataloader import build_dataframe, split_dataframe, get_dataloaders
from model.models import build_model
from helper.metrics import SegmentationMetrics


def auto_detect_dataset_root(configured_path: str) -> str:
    """
    Check if the configured dataset path exists.
    If not, search /kaggle/input and subdirectories to find where
    the 'train' directory with sat images resides.
    """
    if os.path.exists(configured_path) and os.path.exists(os.path.join(configured_path, "train")):
        # Path is valid as-is
        return configured_path
        
    print("[evaluate] Configured DATA_ROOT not found. Auto-detecting dataset root...")
    # Search common paths
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs:
            train_path = Path(root) / "train"
            # Verify it contains satellite images
            sat_files = list(train_path.glob("*_sat.jpg"))
            if len(sat_files) > 0:
                print(f"[evaluate] Detected dataset root: {root}")
                return root
                
    # Fallback to current configuration
    return configured_path


def main():
    device = CFG["DEVICE"]
    
    # ── 1. Determine Checkpoint Path ──────────────────────────────
    # Check if a specific checkpoint path was passed as command line argument
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
    else:
        ckpt_path = Path(CFG["BEST_MODEL"])

    print("=" * 60)
    print("  DeepGlobe Model Evaluation — Per-Class Detailed Metrics")
    print("=" * 60)
    print(f"  Target Device : {device}")
    
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at: {ckpt_path.resolve()}")
        print("   If you are running on Kaggle, please check where your .pth file is saved.")
        return

    print(f"  Checkpoint    : {ckpt_path.name}")
    print(f"  Full Path     : {ckpt_path.resolve()}")
    print("-" * 60)

    # ── 2. Load Checkpoint & Build Model ──────────────────────────
    print("[evaluate] Loading weights...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract saved config if available
    saved_cfg = checkpoint.get("cfg", CFG)
    
    model = build_model(saved_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    print(f"[evaluate] Model successfully loaded (epoch {checkpoint.get('epoch', 'N/A')})")
    
    # ── 3. Load Validation Dataset ────────────────────────────────
    # Resolve dataset root path dynamically
    data_root = auto_detect_dataset_root(CFG["DATA_ROOT"])
    saved_cfg["DATA_ROOT"] = data_root

    print("[evaluate] Loading dataset splits...")
    df = build_dataframe(data_root)
    if len(df) == 0:
        print("ERROR: No images found. Check your dataset configuration.")
        return
        
    train_df, val_df = split_dataframe(df, saved_cfg["VAL_SPLIT"], saved_cfg["SEED"])
    _, val_loader = get_dataloaders(train_df, val_df, saved_cfg)
    print(f"[evaluate] Validation patches to process: {len(val_loader) * saved_cfg['BATCH_SIZE']}")

    # ── 4. Accumulate Predictions ─────────────────────────────────
    metrics = SegmentationMetrics(num_classes=saved_cfg["NUM_CLASSES"])
    
    print("\n[evaluate] Running inference over validation set...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating", bar_format="{l_bar}{bar:20}{r_bar}"):
            images = images.to(device)
            if saved_cfg["TASK"] == "road":
                masks = masks.to(device).float()
            else:
                masks = masks.to(device).long()
                
            logits = model(images)
            metrics.update(logits, masks)

    # ── 5. Calculate Metrics ──────────────────────────────────────
    iou_per_class = metrics._iou_per_class()
    f1_per_class = metrics._f1_per_class()
    pixel_acc = metrics.compute_pixel_accuracy()

    # 7-Class scores (Standard baseline, includes "Unknown" at index 6)
    miou_7class = np.nanmean(iou_per_class)
    f1_7class = np.nanmean(f1_per_class)

    # 6-Class scores (Standard academic baseline, excludes "Unknown" at index 6)
    miou_6class = np.nanmean(iou_per_class[:6])
    f1_6class = np.nanmean(f1_per_class[:6])

    # ── 6. Print Visual Report ────────────────────────────────────
    bar_len = 20
    divider = "=" * 65
    sub_divider = "-" * 65

    print("\n" + divider)
    print(f"  {'Class Name':<18}  {'IoU':>6}  {'F1':>6}  IoU visual bar")
    print(sub_divider)
    for i in range(saved_cfg["NUM_CLASSES"]):
        class_name = LC_CLASSES[i][0]
        iou = iou_per_class[i]
        f1 = f1_per_class[i]
        
        # Build text bar chart
        if not np.isnan(iou):
            bar = "█" * int(iou * bar_len)
        else:
            bar = "N/A"
            
        print(f"  {class_name:<18}  {iou:>6.4f}  {f1:>6.4f}  {bar}")
        
    print(divider)
    print(f"  Overall Pixel Accuracy             : {pixel_acc:.4f}  ({pixel_acc*100:.2f}%)")
    print(sub_divider)
    print(f"  7-Class mIoU (with Unknown)        : {miou_7class:.4f}")
    print(f"  7-Class Mean F1 (with Unknown)     : {f1_7class:.4f}")
    print(sub_divider)
    print(f"  6-Class mIoU (excluding Unknown)   : {miou_6class:.4f}")
    print(f"  6-Class Mean F1 (excluding Unknown): {f1_6class:.4f}")
    print(divider + "\n")


if __name__ == "__main__":
    main()
