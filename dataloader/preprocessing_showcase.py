# =============================================================
#  preprocessing_showcase.py
#
#  Output:
#      dataloader/images/preprocessing_intensity_filters.png
#      dataloader/images/preprocessing_edges_augmentations.png
#      dataloader/images/preprocessing_labels.png
#
# The classical methods (Intensity, Filters, Edges) in this file are ONLY for
# theoretical visualization and data analysis (EDA), and should NOT be applied
# to the main training pipeline (dataloader.py).
#
# - Contrast processing: Do not use HE/CLAHE as it easily causes color noise in the spectrum of
# satellite images. Instead, use Z-score normalization (ImageNet Normalization).
#
# - Spatial filters: Fine filters blur small boundary details (roads,
# land borders) which are extremely important for the Segmentation model.
#
# - Edge finding: Let the SwinFAN model learn edge filters through
# end-to-end training; directly loading edge images loses
# important color and texture information.
# =============================================================

import sys
import os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG, LC_CLASSES, LC_COLOR_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD
from dataloader.dataloader import build_dataframe, rgb_mask_to_index, index_to_rgb

# ─────────────────────────────────────────────────────────────
# Theme constants
# ─────────────────────────────────────────────────────────────
DARK_BG      = "#080814"
PANEL_BG     = "#0f0f22"
PANEL_BG2    = "#12122a"
ACCENT_CYAN  = "#00e5ff"
ACCENT_GOLD  = "#ffd700"
ACCENT_GREEN = "#39ff14"
ACCENT_PINK  = "#ff4d94"
ACCENT_ORANGE= "#ff8c00"
TEXT_WHITE   = "#f0f0ff"
TEXT_MUTED   = "#8888aa"
BORDER_CLR   = "#1e1e40"


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def load_sample(img_path: str, mask_path: str, crop_size: int = 512):
    """Load and centre-crop to crop_size × crop_size."""
    img  = cv2.imread(img_path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    y0 = (h - crop_size) // 2
    x0 = (w - crop_size) // 2
    img  = img [y0:y0+crop_size, x0:x0+crop_size]
    mask = mask[y0:y0+crop_size, x0:x0+crop_size]
    return img, mask


def ax_style(ax, title, accent=ACCENT_CYAN, hide_ticks=True):
    """Apply unified dark theme to an axes."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, fontsize=9.5, color=TEXT_WHITE,
                 fontweight="bold", pad=5, loc="center")
    if hide_ticks:
        ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(accent)
        spine.set_linewidth(0.8)


def plot_histogram(ax, img_rgb, title):
    """Plot RGB + grayscale histogram on ax (axes must NOT be off)."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, fontsize=9.5, color=TEXT_WHITE,
                 fontweight="bold", pad=5, loc="center")
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bins  = np.arange(256)
    colors = [("#ff4d4d", "R"), ("#44ff44", "G"), ("#4d88ff", "B"), ("#cccccc", "Gray")]
    channels = [img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2], gray]
    for ch, (clr, lbl) in zip(channels, colors):
        hist = cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.max()
        ax.plot(bins, hist, color=clr, linewidth=1.0, alpha=0.85, label=lbl)
        ax.fill_between(bins, hist, alpha=0.08, color=clr)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Pixel Value", fontsize=7, color=TEXT_MUTED)
    ax.set_ylabel("Normalized Freq.", fontsize=7, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    ax.grid(True, color=BORDER_CLR, linestyle="--", alpha=0.5)
    ax.legend(fontsize=7, facecolor=PANEL_BG2,
              edgecolor=ACCENT_CYAN, labelcolor=TEXT_WHITE, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(ACCENT_GOLD)
        spine.set_linewidth(0.8)


def annotate_rms(ax, img_rgb):
    """Overlay RMS contrast value on image panel."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    rms  = np.std(gray.astype(np.float32) / 255.0)
    tag  = f"RMS={rms:.3f}"
    clr  = ACCENT_GREEN if rms > 0.18 else (ACCENT_GOLD if rms > 0.12 else ACCENT_PINK)
    ax.text(6, 20, tag, fontsize=8, color=clr, fontweight="bold",
            bbox=dict(facecolor="#000000cc", edgecolor=clr, boxstyle="round,pad=0.2"))


# ─────────────────────────────────────────────────────────────
# PAGE 1: Xử lý Cường độ & Bộ lọc Không gian
# ─────────────────────────────────────────────────────────────
def make_page1(img, mask, out_path: Path):
    """
    Layout (5 rows × 6 cols):
      Row 0  : Section header labels
      Row 1-2: Intensity / Contrast processing
      Row 3-4: Spatial Filters (Smoothing & Sharpening)
    """
    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    fig.suptitle(
        "Phân tích & Tiền xử lý Ảnh Vệ tinh DeepGlobe (Nhóm 24)\n"
        "Khảo sát Lý thuyết (Không dùng trong Training)  vs  Deep Learning Normalization",
        fontsize=15, fontweight="bold", color=ACCENT_CYAN, y=0.98
    )

    # 4 rows × 6 cols, with an extra thin row for section labels
    gs = gridspec.GridSpec(4, 6, figure=fig,
                           hspace=0.42, wspace=0.18,
                           left=0.04, right=0.97,
                           top=0.92, bottom=0.04)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── Label banner: Cường độ & Tương phản ──────────────────
    ax_lbl1 = fig.add_subplot(gs[0, :])
    ax_lbl1.set_facecolor("#0d0d2e")
    ax_lbl1.text(0.5, 0.55,
                 "◆  XỬ LÝ CƯỜNG ĐỘ & TƯƠNG PHẢN (CHỈ KHẢO SÁT LÝ THUYẾT, KHÔNG DÙNG TRONG TRAINING)",
                 transform=ax_lbl1.transAxes, ha="center", va="center",
                 fontsize=10.5, fontweight="bold", color=ACCENT_GOLD)
    ax_lbl1.axis("off")
    for sp in ax_lbl1.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_GOLD); sp.set_linewidth(1.5)

    # ── ROW 1: Intensity panels ─────────────────────────────────
    panels_intensity = []

    # 1. Ảnh gốc + histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(img); annotate_rms(ax, img)
    ax_style(ax, "① Ảnh gốc (RGB Satellite)")
    panels_intensity.append(ax)

    # 2. Ảnh xám
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(gray, cmap="gray")
    ax_style(ax, "② Ảnh xám (Grayscale)", ACCENT_GOLD)

    # 3. Histogram gốc
    ax = fig.add_subplot(gs[1, 2])
    plot_histogram(ax, img, "③ Histogram RGB + Gray (gốc)")

    # 4. Linear Contrast Stretching (Co giãn tuyến tính)
    img_min, img_max = gray.min(), gray.max()
    stretched = ((gray.astype(np.float32) - img_min) /
                 (img_max - img_min + 1e-6) * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(stretched, cmap="gray")
    ax_style(ax, "④ Linear Contrast Stretch\n(co giãn dải tuyến tính)", ACCENT_GREEN)
    ax.text(6, 38, f"min={img_min}  max={img_max}", fontsize=7.5,
            color=ACCENT_GREEN,
            bbox=dict(facecolor="#00000099", edgecolor=ACCENT_GREEN, boxstyle="round,pad=0.2"))

    # 5. Histogram Equalization (cân bằng HE)
    he = cv2.equalizeHist(gray)
    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(he, cmap="gray")
    ax_style(ax, "⑤ Histogram Equalization (HE)\n(cân bằng toàn cục)", ACCENT_PINK)

    # 6. CLAHE (Adaptive HE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    ax = fig.add_subplot(gs[1, 5])
    ax.imshow(clahe_img, cmap="gray")
    ax_style(ax, "⑥ CLAHE (Adaptive HE)\n(cân bằng cục bộ – tốt hơn HE)", ACCENT_ORANGE)

    # Row 2: Histograms so sánh + Normalisation
    ax = fig.add_subplot(gs[2, 0])
    plot_histogram(ax, img, "⑦ Histogram sau Contrast Stretch")
    stretched_rgb = np.stack([stretched]*3, axis=-1)
    # actually plot stretched histogram
    gray_s = stretched
    bins = np.arange(256)
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.set_title("⑦ Hist so sánh: gốc vs Stretched",
                 fontsize=9, color=TEXT_WHITE, fontweight="bold", pad=5)
    h0 = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    h1 = cv2.calcHist([gray_s], [0], None, [256], [0, 256]).flatten()
    ax.fill_between(bins, h0/h0.max(), alpha=0.35, color="#4d88ff", label="Gốc")
    ax.plot(bins, h0/h0.max(), color="#4d88ff", linewidth=1)
    ax.fill_between(bins, h1/h1.max(), alpha=0.35, color=ACCENT_GREEN, label="Stretched")
    ax.plot(bins, h1/h1.max(), color=ACCENT_GREEN, linewidth=1)
    ax.set_xlim(0, 255); ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    ax.grid(True, color=BORDER_CLR, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, facecolor=PANEL_BG2, edgecolor=ACCENT_CYAN, labelcolor=TEXT_WHITE)
    ax.set_xlabel("Pixel Value", fontsize=7, color=TEXT_MUTED)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_GOLD); sp.set_linewidth(0.8)

    ax = fig.add_subplot(gs[2, 1])
    ax.cla(); ax.set_facecolor(PANEL_BG)
    ax.set_title("⑧ Hist: gốc vs HE vs CLAHE",
                 fontsize=9, color=TEXT_WHITE, fontweight="bold", pad=5)
    hhe   = cv2.calcHist([he], [0], None, [256], [0, 256]).flatten()
    hcl   = cv2.calcHist([clahe_img], [0], None, [256], [0, 256]).flatten()
    ax.fill_between(bins, h0/h0.max(), alpha=0.25, color="#4d88ff", label="Gốc")
    ax.plot(bins, h0/h0.max(), color="#4d88ff", linewidth=1)
    ax.fill_between(bins, hhe/hhe.max(), alpha=0.25, color=ACCENT_PINK, label="HE")
    ax.plot(bins, hhe/hhe.max(), color=ACCENT_PINK, linewidth=1)
    ax.fill_between(bins, hcl/hcl.max(), alpha=0.25, color=ACCENT_ORANGE, label="CLAHE")
    ax.plot(bins, hcl/hcl.max(), color=ACCENT_ORANGE, linewidth=1)
    ax.set_xlim(0, 255); ax.tick_params(colors=TEXT_MUTED, labelsize=7)
    ax.grid(True, color=BORDER_CLR, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, facecolor=PANEL_BG2, edgecolor=ACCENT_CYAN, labelcolor=TEXT_WHITE)
    ax.set_xlabel("Pixel Value", fontsize=7, color=TEXT_MUTED)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_GOLD); sp.set_linewidth(0.8)

    # ImageNet Normalization (Deep Learning Standard)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    norm_float = (img.astype(np.float32) / 255.0 - mean) / std
    # Visualise by re-scaling back to [0,1]
    norm_vis = (norm_float - norm_float.min()) / (norm_float.max() - norm_float.min() + 1e-6)

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(norm_vis)
    ax_style(ax, "[9] ImageNet Normalization\n(DL - chuan hoa theo mean/std)", ACCENT_CYAN)
    ax.text(6, 38, f"mean={IMAGENET_MEAN}  std={IMAGENET_STD}",
            fontsize=6.5, color=ACCENT_CYAN,
            bbox=dict(facecolor="#00000099", edgecolor=ACCENT_CYAN, boxstyle="round,pad=0.2"))

    # Gamma Correction gamma = 0.5
    gamma     = 0.5
    lut       = (np.arange(256) / 255.0) ** gamma * 255
    gamma_img = cv2.LUT(gray, lut.astype(np.uint8))
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(gamma_img, cmap="gray")
    ax_style(ax, f"[10] Gamma Correction (y={gamma})\n(tang sang vung toi)", ACCENT_GREEN)

    # Negative Image
    negative = 255 - gray
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(negative, cmap="gray")
    ax_style(ax, "[11] Negative Image\n(dao nguoc cuong do sang)", ACCENT_PINK)

    # Log Transformation
    gray_max = int(np.max(gray))
    if gray_max > 0:
        c       = 255.0 / np.log(1 + gray_max)
        log_img = np.clip(c * np.log(1.0 + gray.astype(np.float32)), 0, 255).astype(np.uint8)
    else:
        log_img = gray.copy()
    ax = fig.add_subplot(gs[2, 5])
    ax.imshow(log_img, cmap="gray")
    ax_style(ax, "[12] Log Transformation\n(nen dai sang - mo rong vung toi)", ACCENT_ORANGE)

    # ── Label banner: Bộ lọc không gian ──────────────────────
    ax_lbl2 = fig.add_subplot(gs[3, :6])
    # Split into two rows: one for the banner and one for the panels
    # We'll reuse this row for panels; add sub-gridspec
    ax_lbl2.set_facecolor("#0d2210")
    ax_lbl2.text(0.5, 0.88,
                 "◆  BỘ LỌC KHÔNG GIAN: LÀM MỊN & LÀM SẮC NÉT (CHỈ KHẢO SÁT LÝ THUYẾT, KHÔNG DÙNG TRONG TRAINING)",
                 transform=ax_lbl2.transAxes, ha="center", va="top",
                 fontsize=10, fontweight="bold", color=ACCENT_GREEN)
    ax_lbl2.axis("off")
    for sp in ax_lbl2.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_GREEN); sp.set_linewidth(1.5)

    sub_gs3 = gridspec.GridSpecFromSubplotSpec(
        1, 6, subplot_spec=gs[3, :], wspace=0.18)

    # Box / Mean Filter (3×3, 7×7)
    box3 = cv2.blur(gray, (3, 3))
    ax = fig.add_subplot(sub_gs3[0])
    ax.imshow(box3, cmap="gray")
    ax_style(ax, "[13] Box/Mean Filter (3x3)\n(loc trung binh - lam mo nhe)", ACCENT_GREEN)

    box7 = cv2.blur(gray, (7, 7))
    ax = fig.add_subplot(sub_gs3[1])
    ax.imshow(box7, cmap="gray")
    ax_style(ax, "[14] Box/Mean Filter (7x7)\n(loc trung binh - lam mo manh)", ACCENT_GREEN)

    # Gaussian Filter
    gauss = cv2.GaussianBlur(gray, (7, 7), 2.0)
    ax = fig.add_subplot(sub_gs3[2])
    ax.imshow(gauss, cmap="gray")
    ax_style(ax, "[15] Gaussian Blur (sigma=2)\n(bo loc Gauss - giam nhieu chuan)", ACCENT_CYAN)

    # Median Filter
    median = cv2.medianBlur(gray, 7)
    ax = fig.add_subplot(sub_gs3[3])
    ax.imshow(median, cmap="gray")
    ax_style(ax, "[16] Median Filter (7x7)\n(bao toan canh - chong nhieu salt&pepper)", ACCENT_GOLD)

    # Unsharp Masking
    blurred   = cv2.GaussianBlur(gray, (9, 9), 1.5)
    unsharp   = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    ax = fig.add_subplot(sub_gs3[4])
    ax.imshow(unsharp, cmap="gray")
    ax_style(ax, "[17] Unsharp Masking\n(lam sac net chi tiet - edge enhance)", ACCENT_PINK)

    # Bilateral Filter
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    ax = fig.add_subplot(sub_gs3[5])
    ax.imshow(bilateral, cmap="gray")
    ax_style(ax, "[18] Bilateral Filter\n(lam min BAO TOAN canh - tot nhat)", ACCENT_ORANGE)

    fig.savefig(str(out_path), dpi=140, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"[SAVED] Page 1 → {out_path}")


# ─────────────────────────────────────────────────────────────
# PAGE 2: Phát hiện Cạnh & Pipeline Tăng cường DL
# ─────────────────────────────────────────────────────────────
def make_page2(img, mask, out_path: Path):
    fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
    fig.suptitle(
        "Phân tích & Tiền xử lý Ảnh Vệ tinh DeepGlobe (Nhóm 24)\n"
        "Khảo sát Lý thuyết Phát hiện Cạnh (Không dùng trong Training)  vs  Deep Learning Augmentation Pipeline (Áp dụng thực tế)",
        fontsize=15, fontweight="bold", color=ACCENT_CYAN, y=0.98
    )

    gs = gridspec.GridSpec(4, 6, figure=fig,
                           hspace=0.42, wspace=0.18,
                           left=0.04, right=0.97,
                           top=0.92, bottom=0.04)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ── Label banner: Phát hiện cạnh ─────────────────────────
    ax_lbl = fig.add_subplot(gs[0, :])
    ax_lbl.set_facecolor("#1a0d2e")
    ax_lbl.text(0.5, 0.55,
                "◆  PHÁT HIỆN CẠNH & ĐẶC TRƯNG (CHỈ KHẢO SÁT LÝ THUYẾT, KHÔNG DÙNG TRONG TRAINING)",
                transform=ax_lbl.transAxes, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=ACCENT_PINK)
    ax_lbl.axis("off")
    for sp in ax_lbl.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_PINK); sp.set_linewidth(1.5)

    # ── ROW 1: Gradient-based edge detectors ─────────────────
    # 1. Original
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(img)
    ax_style(ax, "① Ảnh gốc (RGB)")

    # 2. Sobel X
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / sobel_x.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(sobel_x, cmap="hot")
    ax_style(ax, "[2] Sobel Gradient X (Gx)\n(phat hien canh doc)", ACCENT_PINK)

    # 3. Sobel Y
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = np.abs(sobel_y)
    sobel_y = (sobel_y / sobel_y.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(sobel_y, cmap="hot")
    ax_style(ax, "[3] Sobel Gradient Y (Gy)\n(phat hien canh ngang)", ACCENT_PINK)

    # 4. Sobel magnitude
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(mag, cmap="inferno")
    ax_style(ax, "[4] Sobel Gradient Magnitude\n|G| = sqrt(Gx^2 + Gy^2)", ACCENT_ORANGE)

    # 5. Prewitt
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
    ky = np.array([[-1,-1,-1],[0,0,0],[1,1,1]],  np.float32)
    px = cv2.filter2D(gray.astype(np.float32), -1, kx)
    py = cv2.filter2D(gray.astype(np.float32), -1, ky)
    prewitt = np.sqrt(px**2 + py**2)
    prewitt = (prewitt / prewitt.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 4])
    ax.imshow(prewitt, cmap="inferno")
    ax_style(ax, "[5] Prewitt Filter\n(canh tong quat - nhay hon Sobel deu)", ACCENT_GOLD)

    # 6. Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = np.abs(lap)
    lap = (lap / lap.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[1, 5])
    ax.imshow(lap, cmap="plasma")
    ax_style(ax, "[6] Laplacian (nabla^2 f)\n(phat hien canh moi huong)", ACCENT_CYAN)

    # ── ROW 2: Advanced edge  +  Canny + Gradient direction ──
    # 7. LoG (Laplacian of Gaussian)
    blur_log = cv2.GaussianBlur(gray, (5, 5), 1.0)
    log_img  = cv2.Laplacian(blur_log, cv2.CV_64F)
    log_img  = np.abs(log_img)
    log_img  = (log_img / log_img.max() * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(log_img, cmap="plasma")
    ax_style(ax, "[7] LoG (Laplacian of Gaussian)\n(khu nhieu truoc roi tim canh)", ACCENT_CYAN)

    # 8. Canny low threshold
    canny1 = cv2.Canny(gray, 30,  90)
    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(canny1, cmap="gray")
    ax_style(ax, "[8] Canny (low=30 / high=90)\n(nguong thap - nhieu chi tiet)", ACCENT_GREEN)

    canny2 = cv2.Canny(gray, 80, 200)
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(canny2, cmap="gray")
    ax_style(ax, "[9] Canny (low=80 / high=200)\n(nguong cao - canh chinh xac)", ACCENT_GREEN)

    # 9. Gradient direction (Sobel angle map)
    angle = np.arctan2(sy, sx) * 180.0 / np.pi
    angle_norm = ((angle + 180.0) / 360.0 * 255).astype(np.uint8)
    ax = fig.add_subplot(gs[2, 3])
    ax.imshow(angle_norm, cmap="hsv")
    ax_style(ax, "[10] Gradient Direction Map\ntheta = arctan(Gy/Gx)  [Hue = goc]", ACCENT_GOLD)

    # 10. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    ax = fig.add_subplot(gs[2, 4])
    ax.imshow(morph_grad, cmap="hot")
    ax_style(ax, "[11] Morphological Gradient\n(dilate - erode) -> duong vien", ACCENT_PINK)

    # 11. Canny overlay on original
    overlay = img.copy()
    overlay[canny2 > 0] = [255, 80, 0]
    ax = fig.add_subplot(gs[2, 5])
    ax.imshow(overlay)
    ax_style(ax, "[12] Canny Overlay on RGB\n(canh to mau cam tren anh goc)", ACCENT_ORANGE)

    # ── DL Augmentation banner ───────────────────────────────
    ax_lbl2 = fig.add_subplot(gs[3, :])
    ax_lbl2.set_facecolor("#0a1a10")
    ax_lbl2.text(0.5, 0.86,
                 "◆  DEEP LEARNING – PIPELINE TĂNG CƯỜNG DỮ LIỆU (ÁP DỤNG TRONG TRAINING THỰC TẾ)",
                 transform=ax_lbl2.transAxes, ha="center", va="top",
                 fontsize=10.5, fontweight="bold", color=ACCENT_GREEN)
    ax_lbl2.axis("off")
    for sp in ax_lbl2.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_GREEN); sp.set_linewidth(1.5)

    sub_gs = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[3, :], wspace=0.18)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)

    # 1. Horizontal Flip
    aug1 = A.HorizontalFlip(p=1.0)
    flipped = aug1(image=img)["image"]
    ax = fig.add_subplot(sub_gs[0])
    ax.imshow(flipped)
    ax_style(ax, "[13] HorizontalFlip (p=0.5)\n(lat trai-phai)", ACCENT_GREEN)

    # 2. Random Rotate 90
    aug2 = A.RandomRotate90(p=1.0)
    rotated = aug2(image=img)["image"]
    ax = fig.add_subplot(sub_gs[1])
    ax.imshow(rotated)
    ax_style(ax, "[14] RandomRotate90\n(xoay 0/90/180/270 do)", ACCENT_GREEN)

    # 3. ShiftScaleRotate
    aug3 = A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15,
                               rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=1.0)
    affine = aug3(image=img)["image"]
    ax = fig.add_subplot(sub_gs[2])
    ax.imshow(affine)
    ax_style(ax, "[15] ShiftScaleRotate\n(bien doi affine ngau nhien)", ACCENT_CYAN)

    # 4. Color Jitter
    aug4 = A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.15, p=1.0)
    jittered = aug4(image=img)["image"]
    ax = fig.add_subplot(sub_gs[3])
    ax.imshow(jittered)
    ax_style(ax, "[16] ColorJitter\n(thay doi do sang/mau/tuong phan)", ACCENT_GOLD)

    # 5. Full pipeline (all augmentations combined)
    full_pipe = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
    ])
    np.random.seed(1234)
    full_out = full_pipe(image=img)["image"]
    ax = fig.add_subplot(sub_gs[4])
    ax.imshow(full_out)
    ax_style(ax, "[17] Full Augmentation Pipeline\n(tat ca phep bien doi ket hop)", ACCENT_PINK)

    # 6. Final Normalized Tensor (visualised)
    norm_pipe = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    normed = norm_pipe(image=full_out)["image"]   # float32 HxWx3
    vis    = (normed - normed.min()) / (normed.max() - normed.min() + 1e-6)
    ax = fig.add_subplot(sub_gs[5])
    ax.imshow(vis)
    ax_style(ax, "[18] Tensor cuoi (ImageNet Norm)\nInput vao mo hinh SwinFAN", ACCENT_ORANGE)
    ax.text(6, 36, "-> SwinFAN Model", fontsize=8, color=ACCENT_ORANGE, fontweight="bold",
            bbox=dict(facecolor="#00000099", edgecolor=ACCENT_ORANGE, boxstyle="round,pad=0.2"))

    fig.savefig(str(out_path), dpi=140, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"[SAVED] Page 2 → {out_path}")


# ─────────────────────────────────────────────────────────────
# PAGE 3: Mask Pipeline – Ground Truth & Segmentation Labels
# ─────────────────────────────────────────────────────────────
def make_page3(img, mask, out_path: Path):
    """Show how RGB mask → index map → coloured overlay."""
    fig = plt.figure(figsize=(22, 8), facecolor=DARK_BG)
    fig.suptitle(
        "Pipeline Nhãn Phân vùng (Segmentation Label Pipeline)\n"
        "RGB Mask → Class Index Map → Augmented Overlay → DataLoader Output",
        fontsize=14, fontweight="bold", color=ACCENT_CYAN, y=0.98
    )

    gs = gridspec.GridSpec(1, 6, figure=fig,
                           hspace=0.2, wspace=0.18,
                           left=0.03, right=0.97,
                           top=0.88, bottom=0.06)

    # 1. Original sat image
    ax = fig.add_subplot(gs[0])
    ax.imshow(img)
    ax_style(ax, "① Ảnh vệ tinh gốc (Satellite Image)")

    # 2. RGB Mask (ground truth from DeepGlobe)
    ax = fig.add_subplot(gs[1])
    ax.imshow(mask)
    ax_style(ax, "② RGB Mask (ground truth từ DeepGlobe)\nMỗi màu = 1 lớp đất", ACCENT_GOLD)

    # 3. Index map (greyscale where 0-6 = class)
    idx_map = rgb_mask_to_index(mask)
    ax = fig.add_subplot(gs[2])
    im = ax.imshow(idx_map, cmap="tab10", vmin=0, vmax=6)
    ax_style(ax, "③ Class Index Map\nRGB mask → integer 0-6", ACCENT_PINK)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=range(7))

    # 4. Coloured index map (our palette)
    rgb_from_idx = index_to_rgb(idx_map)
    ax = fig.add_subplot(gs[3])
    ax.imshow(rgb_from_idx)
    ax_style(ax, "④ Reconstructed Colour Mask\nindex → RGB palette (dự án)", ACCENT_CYAN)

    # 5. Overlay (alpha blend)
    alpha   = 0.48
    overlay = (img.astype(np.float32) * (1 - alpha) +
               rgb_from_idx.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    ax = fig.add_subplot(gs[4])
    ax.imshow(overlay)
    ax_style(ax, "⑤ Overlay Mask on Image\n(ảnh vệ tinh + mặt nạ nhãn α=0.5)", ACCENT_ORANGE)

    # 6. Legend panel
    ax = fig.add_subplot(gs[5])
    ax.set_facecolor(PANEL_BG2)
    ax.set_title("⑥ Chú giải các lớp đất\n(DeepGlobe Land Cover – 7 classes)",
                 fontsize=9, color=TEXT_WHITE, fontweight="bold", pad=5)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(ACCENT_CYAN); sp.set_linewidth(0.8)

    for cls_idx, (name, color) in LC_CLASSES.items():
        y_pos = 0.90 - cls_idx * 0.12
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.04, y_pos - 0.04), 0.22, 0.10,
            boxstyle="round,pad=0.01",
            facecolor=np.array(color) / 255.0,
            edgecolor="white", linewidth=0.6,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.32, y_pos + 0.01, f"[{cls_idx}] {name}",
                transform=ax.transAxes, fontsize=8.5,
                color=TEXT_WHITE, va="center", fontweight="bold")

    # Legend: colour percentage from the sample
    unique, counts = np.unique(idx_map, return_counts=True)
    total = idx_map.size
    for cls_idx_val, cnt in zip(unique, counts):
        pct = cnt / total * 100
        y_pos = 0.90 - cls_idx_val * 0.12
        ax.text(0.72, y_pos + 0.01, f"{pct:.1f}%",
                transform=ax.transAxes, fontsize=8,
                color=ACCENT_GOLD, va="center")

    fig.savefig(str(out_path), dpi=140, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"[SAVED] Page 3 → {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 85)
    print("  preprocessing_showcase.py — CV Preprocessing Visualizer")
    print("  Nhóm 24 | SOICT | Giảng viên: TS. Trần Nguyên Ngọc")
    print("=" * 85)
    print("  * GIẢI TRÌNH HỌC THUẬT VỀ TIỀN XỬ LÝ (QUAN TRỌNG):")
    print("  - Các kỹ thuật xử lý ảnh cổ điển (Cường độ, Bộ lọc, Biên cạnh) trong file này CHỈ DÙNG ĐỂ KHẢO SÁT LÝ THUYẾT & EDA.")
    print("  - Chúng KHÔNG được áp dụng vào pipeline huấn luyện thực tế (dataloader.py).")
    print("  - Lý do cụ thể:")
    print("    + Xử lý Cường độ (Contrast/HE): Không dùng HE/CLAHE vì dễ gây méo mó phân bố màu tự nhiên.")
    print("      Chuẩn hóa Z-score (ImageNet mean/std) là đủ để ổn định gradient.")
    print("    + Bộ lọc Không gian (Filters): Lọc mịn làm mờ các chi tiết ranh giới nhỏ (đường sá, ranh giới đất)")
    print("      cực kỳ quan trọng đối với bài toán Semantic Segmentation.")
    print("    + Phát hiện Cạnh (Edges): Để mô hình SwinFAN tự học bộ lọc trích xuất biên cạnh từ ảnh gốc;")
    print("      nạp trực tiếp ảnh cạnh làm mất thông tin màu sắc và kết cấu bề mặt.")
    print("  - Các kỹ thuật Deep Learning (Geometric, ColorJitter, Crop, Normalization) được áp dụng.")
    print("=" * 85 + "\n")

    # Find dataset; pick a sample with multiple classes
    df = build_dataframe(CFG["DATA_ROOT"])
    if len(df) == 0:
        print("[ERROR] Không tìm thấy ảnh. Kiểm tra DATA_ROOT trong config.py!")
        return

    # Use image with many classes for best demo
    # Pick image with large mask size (many unique colours = many classes)
    best_idx = 5   # default
    best_score = 0
    for i in range(min(30, len(df))):
        m = cv2.imread(df.iloc[i]["mask_path"])
        if m is None: continue
        m_rgb = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
        crop  = m_rgb[468:980, 468:980]  # centre crop
        idxm  = rgb_mask_to_index(crop)
        score = len(np.unique(idxm))
        if score > best_score:
            best_score = score
            best_idx   = i

    print(f"[INFO] Chọn ảnh idx={best_idx}: {df.iloc[best_idx]['image_path']}")
    print(f"[INFO] Số lớp đất trong crop: {best_score}")

    img, mask = load_sample(
        df.iloc[best_idx]["image_path"],
        df.iloc[best_idx]["mask_path"],
        crop_size=512
    )

    out_dir = Path(__file__).resolve().parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Đang vẽ Trang 1 – Xử lý Cường độ & Bộ lọc...")
    make_page1(img, mask, out_dir / "preprocessing_intensity_filters.png")

    print("[2/3] Đang vẽ Trang 2 – Phát hiện Cạnh & DL Pipeline...")
    make_page2(img, mask, out_dir / "preprocessing_edges_augmentations.png")

    print("[3/3] Đang vẽ Trang 3 – Segmentation Label Pipeline...")
    make_page3(img, mask, out_dir / "preprocessing_labels.png")

    print("\n" + "=" * 85)
    print(f"  ✓ Hoàn thành! Ảnh được lưu tại:")
    print(f"    {out_dir / 'preprocessing_intensity_filters.png'}")
    print(f"    {out_dir / 'preprocessing_edges_augmentations.png'}")
    print(f"    {out_dir / 'preprocessing_labels.png'}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
