# =============================================================
#  train_v3.py — SwinFAN-v3 Training & Validation Loop
#
#  Run with: python scripts/train_v3.py
#    • ARCH = "swinfan_v3" — Swin-v2-Base backbone with
#      SatlasPretrain Aerial pretraining (ICCV 2023, Apache-2.0)
#    • Backbone pretrained on 302M remote sensing labels
#      at 0.5–2 m/pixel aerial imagery (same scale as DeepGlobe)
#    • First run downloads aerial_swinb_si.pth (~503 MB) from
#      HuggingFace and caches to /kaggle/working/weights/
#    • Checkpoint saved to output/best_model_v3.pth
#    • SwinFAN-v2 checkpoint (best_model_v2.pth) is NOT touched
#    • BATCH_SIZE = 4 + GRAD_ACCUM = 2 (effective batch 8)
#    • Loss = SwinFANv2Loss (Focal 30% + Dice 30% + Lovász 40%)
#    • LR = 3e-4, EPOCHS = 50, CosineAnnealingLR
#
#  Project structure:
#      config.py                 — base hyperparameters & class maps
#      dataloader/dataloader.py  — DeepGlobeDataset, get_dataloaders
#      encoder/swin_encoder.py   — SwinV2BaseEncoder (Swin-v2-B + SatlasPretrain)
#      model/models.py           — build_model factory
#      helper/losses.py          — SwinFANv2Loss (Focal+Dice+Lovász)
#      helper/metrics.py         — SegmentationMetrics (IoU, Acc, F1)
# =============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from config import CFG
from dataloader.dataloader import build_dataframe, split_dataframe, get_dataloaders
from model.models          import build_model
from helper.losses         import SwinFANv2Loss
from helper.metrics        import SegmentationMetrics


# ─────────────────────────────────────────────────────────────
# SwinFAN-v3 Configuration Overrides
# ─────────────────────────────────────────────────────────────
# We deep-copy CFG so that config.py defaults remain untouched.
# Running `python scripts/train.py`    always trains SwinFAN-v1.
# Running `python scripts/train_v2.py` always trains SwinFAN-v2.
# Running `python scripts/train_v3.py` always trains SwinFAN-v3.
# ─────────────────────────────────────────────────────────────
GRAD_ACCUM    = 2      # effective batch = BATCH_SIZE × GRAD_ACCUM
V3_EPOCHS     = 50     # same schedule as v2
V3_BATCH_SIZE = 4      # same physical batch as v2 (fits T4 VRAM)
V3_LR         = 3e-4   # same LR as v2

CFG_V3 = copy.deepcopy(CFG)
CFG_V3["ARCH"]       = "swinfan_v3"
CFG_V3["BATCH_SIZE"] = V3_BATCH_SIZE
CFG_V3["EPOCHS"]     = V3_EPOCHS
CFG_V3["LR"]         = V3_LR
CFG_V3["ETA_MIN"]    = 1e-6
CFG_V3["BEST_MODEL"] = str(Path(CFG["OUTPUT_DIR"]) / "best_model_v3.pth")


# ─────────────────────────────────────────────────────────────
# Mixed-precision helpers (identical to train_v2.py)
# ─────────────────────────────────────────────────────────────
def _make_scaler(device: str) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device=device, enabled=(device == "cuda"))


def _autocast(device: str):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# ─────────────────────────────────────────────────────────────
# Task helpers
# ─────────────────────────────────────────────────────────────
def _prepare_targets(masks: torch.Tensor, task: str, device: str) -> torch.Tensor:
    masks = masks.to(device, non_blocking=True)
    if task == "road":
        return masks.float()
    return masks.long()


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# Training — one epoch (with gradient accumulation)
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model:      nn.Module,
                    loader:     torch.utils.data.DataLoader,
                    optimizer:  torch.optim.Optimizer,
                    scheduler:  torch.optim.lr_scheduler._LRScheduler,
                    criterion:  nn.Module,
                    scaler:     torch.amp.GradScaler,
                    device:     str,
                    task:       str,
                    grad_accum: int = GRAD_ACCUM) -> float:
    """
    Run one full training epoch with gradient accumulation.

    Gradient Accumulation:
        Gradients are accumulated over `grad_accum` mini-batches before
        the optimizer takes a step. This makes the effective batch size
        equal to BATCH_SIZE × grad_accum while keeping peak VRAM usage low.

    Returns:
        float: Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="  [Train]", leave=False,
                bar_format="{l_bar}{bar:20}{r_bar}")

    for step, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks  = _prepare_targets(masks, task, device)

        with _autocast(device):
            logits = model(images)
            # Divide loss by grad_accum so the accumulated gradient
            # magnitude is equivalent to a single large-batch step.
            loss = criterion(logits, masks) / grad_accum

        scaler.scale(loss).backward()

        # Only step the optimizer and zero gradients every grad_accum steps
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Track unscaled loss for reporting
        running_loss += loss.item() * grad_accum
        current_lr    = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}",
                         lr=f"{current_lr:.2e}")

    scheduler.step()
    return running_loss / len(loader)


# ─────────────────────────────────────────────────────────────
# Validation — one epoch (reports both 7-class and 6-class mIoU)
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

    Returns:
        Tuple: (val_loss, val_miou_7, val_miou_6, val_acc, val_f1)
            • val_miou_7 — 7-class mIoU (official metric, includes Unknown)
            • val_miou_6 — 6-class mIoU (academic standard, excludes Unknown)
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

    val_loss = running_loss / len(loader)
    results  = metric.compute()

    # ── 7-class mIoU (official, includes Unknown class at index 6) ──
    val_miou_7 = results["miou"]
    val_acc    = results["pixel_acc"]
    val_f1     = results["f1"]

    # ── 6-class mIoU (academic standard, excludes Unknown) ──────────
    iou_per_class = metric._iou_per_class()
    val_miou_6    = float(np.nanmean(iou_per_class[:6])) if num_classes == 7 else val_miou_7

    return val_loss, val_miou_7, val_miou_6, val_acc, val_f1


# ─────────────────────────────────────────────────────────────
# Auto-detect dataset root (same logic as evaluate_model.py)
# ─────────────────────────────────────────────────────────────
def auto_detect_dataset_root(configured_path: str) -> str:
    if os.path.exists(configured_path) and os.path.exists(
            os.path.join(configured_path, "train")):
        return configured_path

    print("[train_v3] Configured DATA_ROOT not found. Auto-detecting...")
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train" in dirs:
            train_path = Path(root) / "train"
            if list(train_path.glob("*_sat.jpg")):
                print(f"[train_v3] Detected dataset root: {root}")
                return root

    return configured_path


# ─────────────────────────────────────────────────────────────
# Main training script
# ─────────────────────────────────────────────────────────────
def main():
    # ── Resolve dataset path ───────────────────────────────────
    data_root = auto_detect_dataset_root(CFG_V3["DATA_ROOT"])
    CFG_V3["DATA_ROOT"] = data_root

    # ── Setup ──────────────────────────────────────────────────
    seed_everything(CFG_V3["SEED"])
    device     = CFG_V3["DEVICE"]
    task       = CFG_V3["TASK"]
    output_dir = Path(CFG_V3["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt  = Path(CFG_V3["BEST_MODEL"])

    print("=" * 60)
    print(" Segmentation — SwinFAN-v3 (SatlasPretrain Aerial Swin-v2-B)")
    print("=" * 60)
    print(f"  Device            : {device}")
    if device == "cuda":
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA              : {torch.version.cuda}")
    print(f"  Task              : {task}  ({CFG_V3['NUM_CLASSES']} classes)")
    print(f"  Epochs            : {CFG_V3['EPOCHS']}")
    print(f"  Batch size        : {CFG_V3['BATCH_SIZE']} x {GRAD_ACCUM} steps"
          f" = {CFG_V3['BATCH_SIZE'] * GRAD_ACCUM} effective")
    print(f"  LR                : {CFG_V3['LR']}")
    print(f"  Loss              : SwinFANv2Loss (Focal 30% + Dice 30% + Lovász 40%)")
    print(f"  Pretrain          : SatlasPretrain Aerial_SwinB_SI (ICCV 2023)")
    print(f"  Checkpoint        : {best_ckpt.name}")
    print("=" * 60)

    # ── Data ───────────────────────────────────────────────────
    print("\n[1/4] Preparing data...")
    df               = build_dataframe(CFG_V3["DATA_ROOT"])
    train_df, val_df = split_dataframe(df, CFG_V3["VAL_SPLIT"], CFG_V3["SEED"])
    train_loader, val_loader = get_dataloaders(train_df, val_df, CFG_V3)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────
    # NOTE: Building the model triggers the SatlasPretrain weight download
    # if not already cached at /kaggle/working/weights/aerial_swinb_si.pth
    print("\n[2/4] Building model (may download ~503 MB SatlasPretrain weights)...")
    model = build_model(CFG_V3).to(device)

    # ── Loss, Optimizer, Scheduler ─────────────────────────────
    print("\n[3/4] Setting up optimizer and loss...")
    criterion = SwinFANv2Loss(
        num_classes=CFG_V3["NUM_CLASSES"],
        focal_weight=0.3,
        dice_weight=0.3,
        lovasz_weight=0.4,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG_V3["LR"],
        weight_decay=CFG_V3["WEIGHT_DECAY"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG_V3["EPOCHS"],
        eta_min=CFG_V3["ETA_MIN"],
    )
    scaler = _make_scaler(device)

    # ── Training Loop ──────────────────────────────────────────
    print("\n[4/4] Starting training...\n")
    best_miou_7 = -1.0
    history = {
        "train_loss": [], "val_loss": [],
        "val_miou_7": [], "val_miou_6": [],
        "val_pixel_acc": [], "val_f1": [],
    }

    for epoch in range(1, CFG_V3["EPOCHS"] + 1):
        print(f"{'-' * 60}")
        print(f"  Epoch [{epoch:02d}/{CFG_V3['EPOCHS']:02d}]"
              f"  |  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"{'-' * 60}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, device, task,
            grad_accum=GRAD_ACCUM,
        )

        # Validate — returns 7-class and 6-class mIoU separately
        val_loss, val_miou_7, val_miou_6, val_acc, val_f1 = validate(
            model, val_loader, criterion, device,
            CFG_V3["NUM_CLASSES"], task,
        )

        # Log history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou_7"].append(val_miou_7)
        history["val_miou_6"].append(val_miou_6)
        history["val_pixel_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Best checkpoint based on 7-class mIoU (official metric)
        is_best     = val_miou_7 > best_miou_7
        best_miou_7 = max(val_miou_7, best_miou_7)
        star        = "  * NEW BEST" if is_best else ""

        print(f"\n  Train Loss             : {train_loss:.4f}")
        print(f"  Val   Loss             : {val_loss:.4f}")
        print(f"  Val   mIoU  (7-class)  : {val_miou_7:.4f}{star}")
        print(f"  Val   mIoU  (6-class)  : {val_miou_6:.4f}  (excl. Unknown)")
        print(f"  Val   Acc              : {val_acc:.4f}")
        print(f"  Val   F1               : {val_f1:.4f}")

        if is_best:
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou":       best_miou_7,
                "cfg":             CFG_V3,
                "history":         history,
            }, best_ckpt)
            print(f"  [SAVED] -> {best_ckpt}")

    print(f"\n{'=' * 60}")
    print(f"  Training Complete!")
    print(f"  Best Val mIoU (7-class) : {best_miou_7:.4f}")
    print(f"  Checkpoint              : {best_ckpt}")
    print(f"{'=' * 60}\n")

    return model, history


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, history = main()
