# =============================================================
#  train.py — Training & Validation Loop
#
#  Run with: python scripts/train.py
#
#  Supports two tasks (configure via config.py):
#    • "land_cover" — 7-class multi-class segmentation
#    • "road"       — binary segmentation (1 class)
#
#  Evaluation metrics tracked each epoch:
#    • Mean IoU (mIoU)        — primary   [SOICT Group 24 spec]
#    • Overall Pixel Accuracy — secondary [SOICT Group 24 spec]
#    • Mean F1-Score          — secondary [SOICT Group 24 spec]
#
#  Project structure:
#      config.py                 — hyperparameters & class maps
#      dataloader/dataloader.py  — DeepGlobeDataset, get_dataloaders
#      encoder/swin_encoder.py   — SwinFAN Swin-T backbone
#      model/models.py           — SwinFANModel, build_model
#      helper/losses.py          — HybridLoss (Focal + Dice)
#      helper/metrics.py         — SegmentationMetrics (IoU, Acc, F1)
#
#  Project: DeepGlobe Land Cover & Road Segmentation
#  Institution: SOICT, Hanoi University of Science and Technology
#  Group: 24 | Supervisor: Dr. Tran Nguyen Ngoc
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CFG
from dataloader.dataloader import build_dataframe, split_dataframe, get_dataloaders
from model.models        import build_model
from helper.losses       import HybridLoss
from helper.metrics      import SegmentationMetrics


# ─────────────────────────────────────────────────────────────
# Mixed-precision helpers (device-aware, no deprecation warnings)
# ─────────────────────────────────────────────────────────────
def _make_scaler(device: str) -> torch.amp.GradScaler:
    """
    Return a GradScaler that is active only when running on CUDA.
    On CPU it becomes a no-op scaler (enabled=False).
    """
    return torch.amp.GradScaler(device=device, enabled=(device == "cuda"))


def _autocast(device: str):
    """
    Return the appropriate autocast context manager:
      • CUDA  → torch.amp.autocast('cuda', dtype=torch.float16)
      • CPU   → contextlib.nullcontext()  (no-op, no warnings)
    """
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────
def _prepare_targets(masks: torch.Tensor, task: str,
                     device: str) -> torch.Tensor:
    """
    Move targets to device and cast to the correct dtype for each task.

    Land Cover (multi-class): LongTensor   (B, H, W)  — class indices
    Road       (binary):      FloatTensor  (B, H, W)  — [0, 1] labels
    """
    masks = masks.to(device, non_blocking=True)
    if task == "road":
        return masks.float()
    return masks.long()


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# Training — one epoch
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model:     nn.Module,
                    loader:    torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    criterion: nn.Module,
                    scaler:    torch.amp.GradScaler,
                    device:    str,
                    task:      str) -> float:
    """
    Run one full training epoch over all batches.

    Mixed-Precision (FP16) Training:
        • _autocast(device) — on CUDA casts ops to float16 for speed.
        • GradScaler        — prevents float16 underflow; no-op on CPU.
        • clip_grad_norm_   — clips gradients for Transformer stability.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="  [Train]", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)   # (B, 3, H, W)
        masks  = _prepare_targets(masks, task, device)  # (B, H, W)

        optimizer.zero_grad()
        with _autocast(device):
            logits = model(images)                      # (B, C, H, W)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        current_lr    = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")

    scheduler.step()
    return running_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# Validation — one epoch
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model:       nn.Module,
             loader:      torch.utils.data.DataLoader,
             criterion:   nn.Module,
             device:      str,
             num_classes: int,
             task:        str) -> tuple:
    """
    Run one full validation epoch.

    Computes three evaluation metrics required by the SOICT Group 24 spec:
        1. Mean IoU (mIoU)          — primary metric
        2. Overall Pixel Accuracy   — secondary metric
        3. Mean F1-Score            — secondary metric

    Returns:
        Tuple[float, float, float, float]:
            (mean_val_loss, mean_iou, pixel_accuracy, mean_f1)
    """
    model.eval()
    running_loss = 0.0
    metric       = SegmentationMetrics(num_classes=num_classes)

    pbar = tqdm(loader, desc="  [Val]  ", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = _prepare_targets(masks, task, device)

        with _autocast(device):
            logits = model(images)
            loss   = criterion(logits, masks)

        running_loss += loss.item()
        metric.update(logits, masks)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    val_loss  = running_loss / len(loader)
    results   = metric.compute()
    val_miou  = results["miou"]
    val_acc   = results["pixel_acc"]
    val_f1    = results["f1"]

    return val_loss, val_miou, val_acc, val_f1


# ─────────────────────────────────────────────────────────────
# Main training script
# ─────────────────────────────────────────────────────────────
def main():
    # ── Setup ─────────────────────────────────────────────────
    seed_everything(CFG["SEED"])
    device     = CFG["DEVICE"]
    task       = CFG["TASK"]
    output_dir = Path(CFG["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = Path(CFG["BEST_MODEL"])

    arch_label = CFG.get("ARCH", "swinfan").upper()

    print("=" * 60)
    print(f"  DeepGlobe Segmentation — {arch_label}")
    print(f"  SOICT Group 24 | Supervisor: Dr. Tran Nguyen Ngoc")
    print("=" * 60)
    print(f"  Device     : {device}")
    if device == "cuda":
        print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA       : {torch.version.cuda}")
    print(f"  Task       : {task}  ({CFG['NUM_CLASSES']} classes)")
    print(f"  Epochs     : {CFG['EPOCHS']}")
    print(f"  Batch size : {CFG['BATCH_SIZE']}")
    print(f"  LR         : {CFG['LR']}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────
    print("\n[1/4] Preparing data...")
    df               = build_dataframe(CFG["DATA_ROOT"])
    train_df, val_df = split_dataframe(df, CFG["VAL_SPLIT"], CFG["SEED"])
    train_loader, val_loader = get_dataloaders(train_df, val_df, CFG)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    print("\n[2/4] Building model...")
    model = build_model(CFG).to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────
    print("\n[3/4] Setting up optimizer and loss...")
    criterion = HybridLoss(
        num_classes=CFG["NUM_CLASSES"],
        focal_weight=CFG["FOCAL_WEIGHT"],
        dice_weight=CFG["DICE_WEIGHT"],
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["LR"],
        weight_decay=CFG["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG["EPOCHS"],
        eta_min=CFG["ETA_MIN"],
    )
    scaler = _make_scaler(device)

    # ── Training Loop ─────────────────────────────────────────
    print("\n[4/4] Starting training...\n")
    best_miou = -1.0
    history   = {
        "train_loss": [], "val_loss": [],
        "val_miou": [], "val_pixel_acc": [], "val_f1": [],
    }

    for epoch in range(1, CFG["EPOCHS"] + 1):
        print(f"{'-' * 60}")
        print(f"  Epoch [{epoch:02d}/{CFG['EPOCHS']:02d}]"
              f"  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'-' * 60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, task
        )

        # Validate — returns all three metrics
        val_loss, val_miou, val_acc, val_f1 = validate(
            model, val_loader, criterion, device,
            CFG["NUM_CLASSES"], task
        )

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_pixel_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        is_best   = val_miou > best_miou
        best_miou = max(val_miou, best_miou)
        star      = "  * NEW BEST" if is_best else ""

        # Print all three metrics each epoch
        print(f"\n  Train Loss    : {train_loss:.4f}")
        print(f"  Val   Loss    : {val_loss:.4f}")
        print(f"  Val   mIoU   : {val_miou:.4f}{star}")
        print(f"  Val   Acc    : {val_acc:.4f}")
        print(f"  Val   F1     : {val_f1:.4f}")

        # Save best checkpoint (keyed by mIoU as primary metric)
        if is_best:
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou":       best_miou,
                "cfg":             CFG,
                "history":         history,
            }, best_ckpt)
            print(f"  [SAVED] → {best_ckpt}")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best Val mIoU : {best_miou:.4f}")
    print(f"  Checkpoint    : {best_ckpt}")
    print(f"{'=' * 60}\n")

    return model, history


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = main()
