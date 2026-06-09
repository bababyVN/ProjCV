# =============================================================
#  check_dataloader.py — DataLoader Visualisation & Diagnostic
#
#  Shows exactly what the dataloader does with your images:
#    1. Raw satellite image + mask (original, uncropped)
#    2. Training crop with full augmentation pipeline
#    3. Validation crop with only normalisation (no augmentation)
#
#  Run from project root:
#      python dataloader/check_dataloader.py
#
#  Output:
#      Printed tensor statistics in the console.
#      Saved figure → output/dataloader_visualization.png
# =============================================================

import sys
import io
# Force UTF-8 on Windows consoles so box-drawing chars print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

import torch

from config import CFG, LC_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from dataloader.dataloader import (
    DeepGlobeDataset,
    build_dataframe,
    split_dataframe,
    index_to_rgb,
)


# ─────────────────────────────────────────────────────────────
# Utility: undo ImageNet normalisation → uint8 RGB for display
# ─────────────────────────────────────────────────────────────
def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalised (3, H, W) float tensor back to a
    displayable uint8 (H, W, 3) RGB image.

    Undoes the ImageNet mean/std normalisation applied by
    albumentations A.Normalize so that the pixel values are
    human-readable again.

    Args:
        tensor: torch.FloatTensor of shape (3, H, W)

    Returns:
        np.ndarray: uint8 RGB image of shape (H, W, 3)
    """
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(3, 1, 1)
    img  = tensor.cpu().numpy() * std + mean   # (3, H, W), float [0,1]
    img  = np.clip(img, 0.0, 1.0)
    img  = (img * 255).astype(np.uint8)
    return img.transpose(1, 2, 0)             # (H, W, 3)


# ─────────────────────────────────────────────────────────────
# Utility: print tensor statistics to console
# ─────────────────────────────────────────────────────────────
def print_tensor_stats(label: str, image_t: torch.Tensor, mask_t: torch.Tensor):
    """Print shape, dtype, value range and class ids for a sample."""
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    print(f"  Image tensor  shape : {tuple(image_t.shape)}")
    print(f"  Image tensor  dtype : {image_t.dtype}")
    print(f"  Image tensor  min   : {image_t.min():.4f}")
    print(f"  Image tensor  max   : {image_t.max():.4f}")
    print(f"  Image tensor  mean  : {image_t.mean():.4f}")
    print(f"  Image tensor  std   : {image_t.std():.4f}")
    print(f"  Mask  tensor  shape : {tuple(mask_t.shape)}")
    print(f"  Mask  tensor  dtype : {mask_t.dtype}")
    unique = mask_t.unique().tolist()
    print(f"  Mask  unique values : {[int(u) for u in unique]}")
    class_names = [LC_CLASSES[int(u)][0] for u in unique if int(u) in LC_CLASSES]
    print(f"  Classes present     : {class_names}")


# ─────────────────────────────────────────────────────────────
# Build datasets
# ─────────────────────────────────────────────────────────────
def build_datasets():
    """Load the DeepGlobe DataFrame and instantiate train/val datasets."""
    print("\n[check_dataloader] Building dataframe…")
    df = build_dataframe(CFG["DATA_ROOT"])
    train_df, val_df = split_dataframe(df, val_split=CFG["VAL_SPLIT"], seed=CFG["SEED"])

    train_ds = DeepGlobeDataset(df=train_df, mode="train", config=CFG)
    val_ds   = DeepGlobeDataset(df=val_df,   mode="val",   config=CFG)

    print(f"[check_dataloader] Train dataset : {len(train_ds):>5} samples")
    print(f"[check_dataloader] Val   dataset : {len(val_ds):>5} patches")
    return train_ds, val_ds, train_df


# ─────────────────────────────────────────────────────────────
# Load the raw (un-preprocessed) image and mask for reference
# ─────────────────────────────────────────────────────────────
def load_raw_pair(row, patch_size: int):
    """
    Read the full satellite image and its RGB mask from disk.
    Resize both to patch_size × patch_size for the preview panel
    (the original 2448×2448 is too large to display neatly).

    Returns (image_rgb, mask_rgb, mask_index) all as uint8 ndarrays.
    """
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_bgr = cv2.imread(row["mask_path"])
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    # Thumbnail for display only
    ps = patch_size
    img_thumb  = cv2.resize(img,      (ps, ps), interpolation=cv2.INTER_AREA)
    mask_thumb = cv2.resize(mask_rgb, (ps, ps), interpolation=cv2.INTER_NEAREST)

    return img_thumb, mask_thumb


# ─────────────────────────────────────────────────────────────
# Build the 3 × 2 visualisation figure
# ─────────────────────────────────────────────────────────────
def make_figure(raw_img, raw_mask,
                train_image_t, train_mask_t,
                val_image_t,   val_mask_t):
    """
    Draw a side-by-side comparison of:

      Row 1 ─ Raw original (resized thumbnail)
      Row 2 ─ Augmented training crop
      Row 3 ─ Normalised validation crop

    Each row shows the satellite image on the left and its
    corresponding class mask on the right.

    Returns the matplotlib Figure.
    """
    DARK_BG      = "#0f0f1a"
    PANEL_BG     = "#1a1a2e"
    ACCENT_CYAN  = "#00d4ff"
    TEXT_WHITE   = "#e8e8f0"
    TEXT_MUTED   = "#9090b0"

    fig = plt.figure(figsize=(16, 14), facecolor=DARK_BG)
    fig.suptitle(
        "DeepGlobe DataLoader — Image Pipeline Inspector",
        fontsize=19, fontweight="bold",
        color=ACCENT_CYAN, y=0.97
    )

    # 3 rows × 2 cols, with extra space for legend at the bottom
    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        hspace=0.38, wspace=0.06,
        left=0.04, right=0.96,
        top=0.93, bottom=0.14,
    )

    row_labels = [
        "① Raw Image (thumbnail)",
        "② Train Crop — full augmentation (flip / rotate / color jitter / normalize)",
        "③ Val Crop  — normalisation only (deterministic sliding-window patch)",
    ]
    col_labels = ["Satellite Image", "Segmentation Mask"]

    panels = [
        # (image_array,           mask_array)
        (raw_img,                  raw_mask),
        (tensor_to_numpy(train_image_t), index_to_rgb(train_mask_t.numpy())),
        (tensor_to_numpy(val_image_t),   index_to_rgb(val_mask_t.numpy())),
    ]

    for row_idx, (img_panel, mask_panel) in enumerate(panels):
        for col_idx, panel in enumerate((img_panel, mask_panel)):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(panel)
            ax.set_facecolor(PANEL_BG)
            ax.axis("off")

            # Row label on the left image only
            if col_idx == 0:
                ax.set_title(
                    row_labels[row_idx],
                    fontsize=10, color=TEXT_WHITE,
                    loc="left", pad=6, fontweight="bold",
                )
            else:
                ax.set_title(
                    col_labels[col_idx],
                    fontsize=9, color=TEXT_MUTED,
                    loc="left", pad=6,
                )

            # Add a subtle border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ACCENT_CYAN)
                spine.set_linewidth(0.8)

    # ── Legend ────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(
            facecolor=np.array(color) / 255.0,
            edgecolor="white",
            linewidth=0.5,
            label=name,
        )
        for _, (name, color) in LC_CLASSES.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(LC_CLASSES),
        fontsize=10,
        frameon=True,
        facecolor="#1e1e38",
        edgecolor=ACCENT_CYAN,
        labelcolor=TEXT_WHITE,
        bbox_to_anchor=(0.5, 0.01),
        handlelength=1.4,
        handleheight=1.0,
    )

    return fig


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  check_dataloader.py — DataLoader Pipeline Inspector")
    print("=" * 60)

    # ── Build datasets ─────────────────────────────────────────
    train_ds, val_ds, train_df = build_datasets()

    # ── Sample item from each split using user-provided index ──
    max_idx = len(train_ds)
    while True:
        try:
            print(f"Enter image index (0 to {max_idx - 1}): ")
            idx = int(input())
            if 0 <= idx < max_idx:
                break
            print(f"Index out of bounds. Please enter a value between 0 and {max_idx - 1}.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    print(f"\n[check_dataloader] Fetching one training sample (idx={idx})...")
    train_image_t, train_mask_t = train_ds[idx]  # random crop + augmentation

    print(f"[check_dataloader] Fetching one validation patch (idx={idx})...")
    val_image_t, val_mask_t = val_ds[idx]         # deterministic sliding-window crop

    # ── Console statistics ─────────────────────────────────────
    print_tensor_stats("TRAIN SAMPLE - augmented random crop",
                       train_image_t, train_mask_t)
    print_tensor_stats("VAL   PATCH  - normalised fixed crop",
                       val_image_t,   val_mask_t)

    # ── Raw thumbnail for row 1 of the figure ─────────────────
    row_selected = train_df.iloc[idx]
    raw_img, raw_mask = load_raw_pair(row_selected, patch_size=CFG["IMG_SIZE"])

    # ── Build and save figure ──────────────────────────────────
    print("\n[check_dataloader] Rendering visualisation...")
    fig = make_figure(
        raw_img,      raw_mask,
        train_image_t, train_mask_t,
        val_image_t,   val_mask_t,
    )

    output_dir = Path(__file__).resolve().parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"dataloader_visualization_{idx}.png"

    fig.savefig(str(save_path), dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[check_dataloader] Figure saved -> {save_path}")
    print("[check_dataloader] Done.\n")

    plt.show()


if __name__ == "__main__":
    main()
