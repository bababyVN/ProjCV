# =============================================================
#  config.py — Global Configuration
#  All hyperparameters and constants live here.
#  Import with: from config import CFG, LC_CLASSES, LC_COLOR_TO_CLASS
# =============================================================

import torch
from pathlib import Path

# Project root = the directory that contains this config.py file
_PROJECT_ROOT = Path(__file__).resolve().parent

# ─────────────────────────────────────────────────────────────
# Main configuration dictionary
# ─────────────────────────────────────────────────────────────
CFG = {
    # ── Task ──────────────────────────────────────────────────
    # "land_cover" → 7-class multi-label segmentation
    # "road"       → binary segmentation (1 class)
    "TASK":          "land_cover",
    "NUM_CLASSES":   7,

    # ── Dataset Paths ──────────────────────────────────────────
    # Local paths — data is downloaded by download_dataset.py into data/train/
    "DATA_ROOT":     str(_PROJECT_ROOT / "data"),
    "OUTPUT_DIR":    str(_PROJECT_ROOT / "output"),
    "BEST_MODEL":    str(_PROJECT_ROOT / "output" / "best_model.pth"),

    # ── Image & Patch Sizes ────────────────────────────────────
    # DeepGlobe full images are 2448×2448 px.
    # We crop them into smaller patches for GPU-efficient training.
    "IMG_SIZE":      512,    # Training patch size (H × W)
    "FULL_SIZE":     2448,   # Original image resolution

    # ── Training Hyperparameters ───────────────────────────────
    "EPOCHS":        30,
    "BATCH_SIZE":    8,      # Reduce to 4 if you get OOM on T4
    "NUM_WORKERS":   2,      # Kaggle allows up to 4 workers
    "SEED":          42,

    # ── Optimizer ─────────────────────────────────────────────
    "LR":            6e-4,
    "WEIGHT_DECAY":  1e-4,
    "ETA_MIN":       1e-6,   # Minimum LR for CosineAnnealingLR

    # ── Model Architecture ─────────────────────────────────────
    # "hybrid" → CustomHybridEncoder (CNN + Transformer) + UNetDecoder
    # You can add more options here as you experiment.
    "ARCH":          "hybrid",

    # ── Loss Weights (FocalLoss + DiceLoss) ───────────────────
    "FOCAL_WEIGHT":  0.5,
    "DICE_WEIGHT":   0.5,
    "FOCAL_GAMMA":   2.0,
    "FOCAL_ALPHA":   0.25,

    # ── Validation ────────────────────────────────────────────
    "VAL_SPLIT":     0.15,   # 15% of data used for validation

    # ── Sliding Window Inference ──────────────────────────────
    # During inference on full 2448×2448 images, we slide a
    # 512×512 window with SW_OVERLAP pixels of overlap between
    # adjacent patches to reduce boundary seam artifacts.
    "SW_PATCH_SIZE": 512,
    "SW_OVERLAP":    64,

    # ── Device ────────────────────────────────────────────────
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}


# ─────────────────────────────────────────────────────────────
# DeepGlobe Land Cover — Class Definitions
# key   : integer class index (0-6)
# value : (class name, RGB colour in the MASK image)
# ─────────────────────────────────────────────────────────────
LC_CLASSES = {
    0: ("Urban land",   (0,   255, 255)),
    1: ("Agriculture",  (255, 255,   0)),
    2: ("Rangeland",    (255,   0, 255)),
    3: ("Forest land",  (0,   255,   0)),
    4: ("Water",        (0,     0, 255)),
    5: ("Barren land",  (255, 255, 255)),
    6: ("Unknown",      (0,     0,   0)),
}

# Reverse lookup: RGB tuple → class index
# Used in dataset.py when converting mask images to index maps.
LC_COLOR_TO_CLASS = {color: idx for idx, (_, color) in LC_CLASSES.items()}


# ─────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# Used in dataset.py augmentation pipelines.
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─────────────────────────────────────────────────────────────
# Quick sanity-check when run directly
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  DeepGlobe Project — Configuration")
    print("=" * 50)
    for k, v in CFG.items():
        print(f"  {k:<20}: {v}")
    print("\nLand Cover Classes:")
    for idx, (name, color) in LC_CLASSES.items():
        print(f"  [{idx}] {name:<15} -> RGB {color}")
