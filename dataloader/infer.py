# =============================================================
#  infer.py — Sliding Window Inference & Visualisation (Complete)
#
#  Run with: python infer.py
# =============================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, LC_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from dataloader.dataloader import rgb_mask_to_index, index_to_rgb, build_dataframe, split_dataframe
from model.models  import build_model


# ─────────────────────────────────────────────────────────────
# Normalisation pipeline (no augmentation, no random crop)
# ─────────────────────────────────────────────────────────────
_INFER_TRANSFORM = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

def preprocess_patch(patch_rgb: np.ndarray) -> torch.Tensor:
    """Normalise a uint8 RGB patch and return a (1, 3, H, W) tensor."""
    return _INFER_TRANSFORM(image=patch_rgb)["image"].unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Sliding Window Inference
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sliding_window_inference(model:       torch.nn.Module,
                             image_path:  str,
                             patch_size:  int = CFG["SW_PATCH_SIZE"],
                             overlap:     int = CFG["SW_OVERLAP"],
                             num_classes: int = CFG["NUM_CLASSES"],
                             device:      str = CFG["DEVICE"]) -> np.ndarray:
    """
    Run segmentation inference on a full-resolution image
    (e.g. 2448 × 2448 px) using an overlapping sliding window.

    ──────── WHY SLIDING WINDOW? ────────────────────────────────

    GPU VRAM cannot fit a 2448×2448 image in a single forward pass.
    We divide it into overlapping 512×512 patches, run each patch
    through the model, and stitch the softmax outputs back together.

    ──────── GAUSSIAN WEIGHTING ─────────────────────────────────

    At the overlapping borders between adjacent patches, a naive
    average would produce visible "seam" artifacts (abrupt class
    changes at patch boundaries). We fix this with a Hanning
    (cosine) window: pixels near the patch centre receive higher
    weight than pixels near the edges, so the contribution fades
    smoothly across overlapping regions.

    Args:
        model       : Trained segmentation model in eval mode.
        image_path  : Path to the full-resolution satellite image.
        patch_size  : H × W of each inference patch (default 512).
        overlap     : Pixels of overlap between adjacent patches (default 64).
        num_classes : Number of segmentation classes.
        device      : "cuda" or "cpu".

    Returns:
        np.ndarray: (H, W) integer class index map for the whole image.
    """
    model.eval()

    # ── Load image ────────────────────────────────────────────
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    H, W  = image.shape[:2]

    # ── Accumulators ──────────────────────────────────────────
    # logit_map  : sum of (softmax probability × weight) per pixel per class
    # weight_map : sum of weights per pixel (for normalisation)
    logit_map  = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # ── Gaussian-like weight kernel (Hanning window) ──────────
    # Pixels at the centre of the patch have weight ≈ 1.0,
    # pixels at the edges have weight ≈ 0.0.
    gy           = np.hanning(patch_size).reshape(-1, 1)   # (H, 1)
    gx           = np.hanning(patch_size).reshape(1, -1)   # (1, W)
    gauss_kernel = (gy * gx).astype(np.float32)             # (H, W)

    # ── Sliding window coordinates ────────────────────────────
    step = patch_size - overlap
    ys   = list(range(0, H - patch_size, step)) + [H - patch_size]
    xs   = list(range(0, W - patch_size, step)) + [W - patch_size]
    ys   = sorted(set(max(0, y) for y in ys))
    xs   = sorted(set(max(0, x) for x in xs))

    total_patches = len(ys) * len(xs)
    pbar = tqdm(total=total_patches, desc="  Inference", unit="patch")

    for y in ys:
        for x in xs:
            # ── Extract patch ──────────────────────────────────
            patch  = image[y : y + patch_size, x : x + patch_size]  # (H, W, 3)
            tensor = preprocess_patch(patch).to(device)              # (1, 3, H, W)

            # ── Forward pass ──────────────────────────────────
            with autocast(enabled=(device == "cuda")):
                logits = model(tensor)                               # (1, C, H', W')

            # Resize back to patch_size if the model changed the resolution
            if logits.shape[-2:] != (patch_size, patch_size):
                logits = F.interpolate(logits, size=(patch_size, patch_size),
                                       mode="bilinear", align_corners=False)

            probs = torch.softmax(logits, dim=1)                     # (1, C, H, W)
            probs = probs.squeeze(0).cpu().numpy()                   # (C, H, W)

            # ── Accumulate weighted softmax probabilities ──────
            logit_map[:, y : y + patch_size, x : x + patch_size] += \
                probs * gauss_kernel[np.newaxis, ...]
            weight_map[y : y + patch_size, x : x + patch_size]   += gauss_kernel

            pbar.update(1)

    pbar.close()

    # ── Normalise by accumulated weights ──────────────────────
    weight_map = np.maximum(weight_map, 1e-8)
    logit_map /= weight_map[np.newaxis, ...]

    # ── Argmax → predicted class index map ────────────────────
    pred_mask = logit_map.argmax(axis=0).astype(np.uint8)     # (H, W)
    return pred_mask


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────
def visualise_prediction(image_path:  str,
                          mask_path:   str,
                          pred_mask:   np.ndarray,
                          save_path:   str = None,
                          num_classes: int = CFG["NUM_CLASSES"]):
    """
    Side-by-side plot:
        [Satellite Image]  |  [Ground Truth Mask]  |  [Predicted Mask]

    Args:
        image_path  : Path to the full-resolution satellite image.
        mask_path   : Path to the ground-truth mask image.
        pred_mask   : (H, W) numpy array of predicted class indices.
        save_path   : If provided, saves the figure to this path.
        num_classes : Number of classes (for legend).
    """
    # ── Load image ────────────────────────────────────────────
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── Decode ground-truth mask ──────────────────────────────
    if num_classes == 1:
        gt_raw  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        gt_rgb  = np.stack([gt_raw] * 3, axis=-1)
        pred_rgb = np.stack([(pred_mask * 255).astype(np.uint8)] * 3, axis=-1)
        legend  = None
    else:
        mask_bgr = cv2.imread(str(mask_path))
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        gt_index = rgb_mask_to_index(mask_rgb)
        gt_rgb   = index_to_rgb(gt_index)
        pred_rgb = index_to_rgb(pred_mask)
        legend   = [
            mpatches.Patch(
                color=np.array(LC_CLASSES[i][1]) / 255.0,
                label=LC_CLASSES[i][0],
            )
            for i in range(num_classes)
        ]

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.patch.set_facecolor("#1a1a2e")

    panel_titles = ["Satellite Image", "Ground Truth", "Prediction"]
    panels       = [image, gt_rgb, pred_rgb]

    for ax, title, panel in zip(axes, panel_titles, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=15, fontweight="bold",
                     color="white", pad=10)
        ax.axis("off")

    if legend:
        fig.legend(
            handles=legend, loc="lower center",
            ncol=num_classes, fontsize=10,
            frameon=True, facecolor="#2d2d44",
            labelcolor="white",
            bbox_to_anchor=(0.5, -0.04),
        )

    plt.suptitle("DeepGlobe — Hybrid CNN + Transformer Segmentation",
                 fontsize=17, fontweight="bold", color="white", y=1.02)
    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[infer] Figure saved -> {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# Compute IoU on a single full-image prediction
# ─────────────────────────────────────────────────────────────
def compute_single_image_iou(pred_mask: np.ndarray,
                              mask_path: str,
                              num_classes: int = CFG["NUM_CLASSES"]) -> dict:
    """
    Compute per-class and mean IoU for one predicted mask
    versus its ground truth.

    Returns:
        dict: {class_name: iou, ..., "mIoU": mean_iou}
    """
    mask_bgr = cv2.imread(str(mask_path))
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    gt_index = rgb_mask_to_index(mask_rgb)

    iou_per_class = {}
    ious = []
    for cls_idx, (cls_name, _) in LC_CLASSES.items():
        pred_c = (pred_mask == cls_idx)
        gt_c   = (gt_index  == cls_idx)
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        iou    = inter / union if union > 0 else float("nan")
        iou_per_class[cls_name] = round(float(iou), 4)
        if union > 0:
            ious.append(iou)

    iou_per_class["mIoU"] = round(float(np.nanmean(ious)), 4)
    return iou_per_class


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = CFG["DEVICE"]
    print(f"[infer] Device: {device}")

    # ── Load best checkpoint ───────────────────────────────────
    ckpt_path = Path(CFG["BEST_MODEL"])
    assert ckpt_path.exists(), \
        f"Checkpoint not found at {ckpt_path}. Run train.py first."

    print(f"[infer] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = build_model(CFG).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"[infer] Model loaded  (epoch {checkpoint['epoch']}"
          f"  |  best mIoU {checkpoint['best_miou']:.4f})")

    # ── Pick a sample image from the validation split ──────────
    df               = build_dataframe(CFG["DATA_ROOT"])
    _, val_df        = split_dataframe(df, CFG["VAL_SPLIT"], CFG["SEED"])
    sample           = val_df.iloc[0]
    print(f"[infer] Running inference on: {sample['image_path']}")

    # ── Sliding window inference ───────────────────────────────
    pred_mask = sliding_window_inference(
        model,
        image_path=sample["image_path"],
        patch_size=CFG["SW_PATCH_SIZE"],
        overlap=CFG["SW_OVERLAP"],
        num_classes=CFG["NUM_CLASSES"],
        device=device,
    )

    # ── Compute and print IoU ──────────────────────────────────
    iou_report = compute_single_image_iou(pred_mask, sample["mask_path"])
    print("\n[infer] IoU Report:")
    for k, v in iou_report.items():
        print(f"  {k:<18}: {v:.4f}")

    # ── Visualise ─────────────────────────────────────────────
    save_path = Path(CFG["OUTPUT_DIR"]) / "sample_prediction.png"
    visualise_prediction(
        image_path=sample["image_path"],
        mask_path=sample["mask_path"],
        pred_mask=pred_mask,
        save_path=str(save_path),
        num_classes=CFG["NUM_CLASSES"],
    )
    