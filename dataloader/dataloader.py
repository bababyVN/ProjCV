# =============================================================
#  dataloader.py — Dataset & DataLoader
#
#  Fully implemented for DeepGlobe Land Cover / Road Extraction.
#  Import with: from dataloader.dataloader import DeepGlobeDataset, get_dataloaders
# =============================================================

import os
import cv2
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, LC_CLASSES, LC_COLOR_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD


# ─────────────────────────────────────────────────────────────
# UTILITY: RGB mask → class index map
# ─────────────────────────────────────────────────────────────
def rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert a 3-channel RGB mask image (H × W × 3, dtype uint8)
    into a 2D integer class index map (H × W, dtype int64).

    DeepGlobe provides segmentation masks as colour-coded RGB images,
    where each pixel colour corresponds to a land cover class.

    LC_COLOR_TO_CLASS (imported from config.py) maps each RGB tuple
    to an integer class index (0–6).

    Args:
        mask_rgb (np.ndarray): Shape (H, W, 3), RGB format.

    Returns:
        np.ndarray: Shape (H, W), dtype int64, values in [0, NUM_CLASSES-1].
    """
    h, w  = mask_rgb.shape[:2]
    index = np.zeros((h, w), dtype=np.int64)

    for color, cls_idx in LC_COLOR_TO_CLASS.items():
        # Build a boolean mask where ALL three channels match this class colour
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        index[match] = cls_idx

    return index


def index_to_rgb(index_mask: np.ndarray) -> np.ndarray:
    """
    Convert a 2D integer class map back to an RGB image for visualisation.

    Args:
        index_mask (np.ndarray): Shape (H, W), dtype int.

    Returns:
        np.ndarray: Shape (H, W, 3), dtype uint8, RGB format.
    """
    h, w = index_mask.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, (_, color) in LC_CLASSES.items():
        rgb[index_mask == cls_idx] = color
    return rgb


# ─────────────────────────────────────────────────────────────
# UTILITY: Build a DataFrame of image/mask file paths
# ─────────────────────────────────────────────────────────────
def build_dataframe(data_root: str) -> pd.DataFrame:
    """
    Scan the DeepGlobe dataset directory and return a DataFrame
    with two columns: 'image_path' and 'mask_path'.

    Expected directory structure on Kaggle:
        <data_root>/
            train/
                000001_sat.jpg   ← satellite image
                000001_mask.png  ← paired segmentation mask
                000002_sat.jpg
                000002_mask.png
                ...

    Args:
        data_root (str): Root path of the dataset, e.g.
            "/kaggle/input/deepglobe-land-cover-classification-dataset"

    Returns:
        pd.DataFrame: Columns ['image_path', 'mask_path'].
    """
    train_dir = Path(data_root) / "train"
    records   = []

    for img_file in sorted(train_dir.glob("*_sat.jpg")):
        stem      = img_file.stem.replace("_sat", "")
        mask_file = train_dir / f"{stem}_mask.png"
        if mask_file.exists():
            records.append({
                "image_path": str(img_file),
                "mask_path":  str(mask_file),
            })

    df = pd.DataFrame(records)
    print(f"[dataset] Found {len(df)} image-mask pairs in: {train_dir}")
    return df


def split_dataframe(df: pd.DataFrame,
                    val_split: float = 0.15,
                    seed: int = 42):
    """Randomly split df into (train_df, val_df)."""
    df       = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val    = int(len(df) * val_split)
    val_df   = df[:n_val].reset_index(drop=True)
    train_df = df[n_val:].reset_index(drop=True)
    print(f"[dataset] Split -> Train: {len(train_df)} | Val: {len(val_df)}")
    return train_df, val_df


# ─────────────────────────────────────────────────────────────
# DeepGlobeDataset
# ─────────────────────────────────────────────────────────────
class DeepGlobeDataset(Dataset):
    """
    PyTorch Dataset for DeepGlobe Land Cover / Road Extraction.

    TRAINING mode  (mode="train"):
        • For each call to __getitem__(idx):
            1. Load the full satellite image and mask from disk.
            2. Random-crop a 512×512 patch (online patch sampling).
               Up to MAX_CROP_RETRIES attempts to skip all-"Unknown" patches.
            3. Convert the cropped mask-RGB to a class index map.
               (Doing this *after* cropping is ~23× faster than full-image.)
            4. Apply albumentations augmentation + ImageNet normalisation.
            5. Return (image_tensor [3,H,W], mask_tensor [H,W]).

    VALIDATION mode (mode="val"):
        • In __init__, pre-generate a list of all (df_idx, y, x) patch
          positions using a non-overlapping sliding window of size IMG_SIZE.
        • __getitem__ deterministically loads only that specific crop, applies
          only normalisation (no random augmentation), and returns tensors.
          This ensures every pixel is evaluated exactly once.
    """

    # How many times to retry a random crop before giving up and
    # accepting an "Unknown"-heavy patch rather than stalling the worker.
    MAX_CROP_RETRIES = 10

    def __init__(self, df: pd.DataFrame, mode: str = "train",
                 config: dict = CFG, transform=None):
        """
        Args:
            df        : DataFrame with 'image_path' and 'mask_path' columns.
            mode      : "train" or "val"
            config    : CFG dictionary from config.py
            transform : Optional albumentations Compose pipeline.
                        If None, the default pipeline for the given mode
                        will be applied.
        """
        self.df         = df.reset_index(drop=True)
        self.mode       = mode
        self.config     = config
        self.patch_size = config["IMG_SIZE"]
        self.full_size  = config.get("FULL_SIZE", 2448)
        self.task       = config.get("TASK", "land_cover")

        # ── Transforms ────────────────────────────────────────
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = get_train_transform(self.patch_size)
        else:
            self.transform = get_val_transform()

        # ── Validation-only: pre-build sliding-window patch list ──
        if self.mode == "val":
            self.patches = []
            ps = self.patch_size

            for idx, row in self.df.iterrows():
                # Read only the header (shape) to avoid loading the full image
                img = cv2.imread(row["image_path"])
                if img is None:
                    H, W = self.full_size, self.full_size
                else:
                    H, W = img.shape[:2]
                    del img  # free memory immediately

                for y in range(0, H - ps + 1, ps):
                    for x in range(0, W - ps + 1, ps):
                        self.patches.append((idx, y, x))

            print(f"[dataset] Val patches: {len(self.patches)} "
                  f"from {len(self.df)} images "
                  f"(grid {self.full_size//ps}×{self.full_size//ps})")

    # ── Length ────────────────────────────────────────────────
    def __len__(self) -> int:
        if self.mode == "val":
            return len(self.patches)
        return len(self.df)

    # ── Single Item ───────────────────────────────────────────
    def __getitem__(self, idx: int):
        """
        Load and return one sample: (image_tensor, mask_tensor).

        image_tensor : torch.FloatTensor  shape (3, H, W)
                       Normalised to ImageNet mean/std.
        mask_tensor  : torch.LongTensor   shape (H, W)  — land_cover
                       torch.FloatTensor  shape (H, W)  — road (binary)
        """
        ps = self.patch_size

        if self.mode == "train":
            row   = self.df.iloc[idx]
            image = self._load_image(row["image_path"])   # (H, W, 3) uint8 RGB
            H, W  = image.shape[:2]

            # Load full mask once outside the loop to avoid redundant disk reads during retries
            if self.task == "land_cover":
                mask_full = cv2.imread(row["mask_path"])
                if mask_full is None:
                    raise FileNotFoundError(f"[dataset] Cannot load mask: {row['mask_path']}")
                mask_full = cv2.cvtColor(mask_full, cv2.COLOR_BGR2RGB)
            else:
                mask_full = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask_full is None:
                    raise FileNotFoundError(f"[dataset] Cannot load mask: {row['mask_path']}")

            # ── Online Patch Sampling with retry ──────────────
            # Retry loop: try to find a patch with at least some labelled
            # pixels. For land_cover we skip fully-"Unknown" (black) patches.
            for _ in range(self.MAX_CROP_RETRIES):
                y = np.random.randint(0, H - ps + 1)
                x = np.random.randint(0, W - ps + 1)

                img_crop  = image[y:y+ps, x:x+ps]           # (ps, ps, 3)

                # For land_cover: check for signal on preloaded mask
                if self.task == "land_cover":
                    mask_crop = mask_full[y:y+ps, x:x+ps]    # (ps, ps, 3)

                    # Skip if the entire crop is the "Unknown" black colour
                    is_all_unknown = np.all(mask_crop == np.array([0, 0, 0],
                                                                   dtype=np.uint8))
                    if is_all_unknown:
                        continue  # retry with a new random location

                    # ── Convert crop to index map (AFTER cropping = 23× faster) ──
                    mask_index = rgb_mask_to_index(mask_crop)  # (ps, ps) int64
                else:
                    # road task: grayscale binary mask crop
                    mask_crop = mask_full[y:y+ps, x:x+ps]
                    mask_index = (mask_crop > 127).astype(np.float32)  # (ps, ps)

                break  # acceptable patch found

            # ── Augmentation ──────────────────────────────────
            augmented = self.transform(image=img_crop, mask=mask_index)
            image_t   = augmented["image"]   # (3, ps, ps) float32 tensor
            mask_t    = augmented["mask"]    # (ps, ps) tensor

            if self.task == "land_cover":
                return image_t, mask_t.long()
            else:
                return image_t, mask_t.float()

        else:  # mode == "val"
            df_idx, y, x = self.patches[idx]
            row = self.df.iloc[df_idx]

            image    = self._load_image(row["image_path"])
            img_crop = image[y:y+ps, x:x+ps]

            if self.task == "land_cover":
                mask_rgb   = cv2.imread(row["mask_path"])
                mask_rgb   = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
                mask_crop  = mask_rgb[y:y+ps, x:x+ps]
                mask_index = rgb_mask_to_index(mask_crop)   # (ps, ps) int64
            else:
                mask_gray  = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
                mask_crop  = mask_gray[y:y+ps, x:x+ps]
                mask_index = (mask_crop > 127).astype(np.float32)

            # Only normalisation, no random spatial augmentation
            augmented = self.transform(image=img_crop, mask=mask_index)
            image_t   = augmented["image"]
            mask_t    = augmented["mask"]

            if self.task == "land_cover":
                return image_t, mask_t.long()
            else:
                return image_t, mask_t.float()

    # ── Internal helpers ──────────────────────────────────────
    def _load_image(self, path: str) -> np.ndarray:
        """
        Load a satellite image from disk.
        Returns a uint8 RGB numpy array of shape (H, W, 3).
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"[dataset] Cannot load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: str) -> np.ndarray:
        """
        Load and decode a full mask image from disk.
        For Land Cover → returns int64 index array (H, W).
        For Road       → returns float32 binary array (H, W).

        Note: For training we use the raw RGB/grayscale crop inside
        __getitem__ to avoid converting the entire 2448×2448 image.
        This helper is kept for external use (e.g. inference or debugging).
        """
        if self.task == "land_cover":
            mask = cv2.imread(path)
            if mask is None:
                raise FileNotFoundError(f"[dataset] Cannot load mask: {path}")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            return rgb_mask_to_index(mask)
        else:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"[dataset] Cannot load mask: {path}")
            return (mask > 127).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────
def get_train_transform(img_size: int) -> A.Compose:
    """
    Return an albumentations Compose pipeline for TRAINING.

    Spatial augmentations are applied consistently to both image and mask.
    Colour augmentations are applied to the image only (albumentations
    handles this correctly when mask is passed alongside).

    Pipeline:
        HorizontalFlip  → mirror left/right  (p=0.5)
        VerticalFlip    → mirror top/bottom  (p=0.5)
        RandomRotate90  → 0°/90°/180°/270°  (p=0.5)
        Transpose       → swap H and W axes  (p=0.3)
        ShiftScaleRotate → small affine jitter  (p=0.4)
        ColorJitter     → brightness/contrast/saturation  (p=0.3)
        Normalize       → ImageNet mean/std
        ToTensorV2      → HWC numpy → CHW torch.Tensor
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """
    Return an albumentations Compose pipeline for VALIDATION.

    No spatial or colour augmentations — only normalisation and tensor
    conversion. The sliding-window crops already provide fixed patches.
    """
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────
def get_dataloaders(train_df: pd.DataFrame,
                    val_df:   pd.DataFrame,
                    config:   dict = CFG):
    """
    Assemble and return (train_loader, val_loader).

    Key DataLoader settings:
        train: shuffle=True, drop_last=True  (stable batch sizes for AMP)
        val:   shuffle=False, drop_last=False (evaluate every patch)
        both:  pin_memory=True (faster CPU→GPU transfer via pinned RAM)
               persistent_workers=True (keep worker processes alive between epochs)

    Args:
        train_df : Training split DataFrame.
        val_df   : Validation split DataFrame.
        config   : CFG from config.py.

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_transform = get_train_transform(config["IMG_SIZE"])
    val_transform   = get_val_transform()

    train_dataset = DeepGlobeDataset(
        df=train_df, mode="train", config=config, transform=train_transform
    )
    val_dataset = DeepGlobeDataset(
        df=val_df, mode="val", config=config, transform=val_transform
    )

    num_workers = config.get("NUM_WORKERS", 2)
    # persistent_workers requires num_workers > 0
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,           # drop incomplete last batch (stable for AMP / BN)
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,          # evaluate every patch exactly once
        persistent_workers=persistent,
    )

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────
# Quick sanity check — run `python -m dataloader.dataloader` to test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("dataloader.py — Running sanity check...")
    df               = build_dataframe(CFG["DATA_ROOT"])
    train_df, val_df = split_dataframe(df)
    print(f"Total samples: {len(df)}")
    print(f"First row:\n{df.iloc[0]}")

    train_loader, val_loader = get_dataloaders(train_df, val_df)
    images, masks = next(iter(train_loader))
    print(f"\n[Train] Batch — images: {images.shape}  dtype: {images.dtype}")
    print(f"[Train] Batch — masks : {masks.shape}   dtype: {masks.dtype}")
    print(f"[Train] Mask  unique values: {masks.unique().tolist()}")

    images_v, masks_v = next(iter(val_loader))
    print(f"\n[Val]   Batch — images: {images_v.shape}  dtype: {images_v.dtype}")
    print(f"[Val]   Batch — masks : {masks_v.shape}   dtype: {masks_v.dtype}")
