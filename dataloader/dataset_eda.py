# =============================================================
#  dataset_eda.py — Dataset Exploratory Data Analysis & Visualisation
#
#  Performs visual analysis:
#    1. Color distribution (RGB histograms with channel-wise means/stds)
#    2. Contrast level analysis (Grayscale intensity histogram, RMS contrast distribution, high/low contrast labeling)
#    3. Class pixel distribution (percentage bar chart colored by category)
#    4. Sample grid showing low, median, and high contrast satellite images
#
#  Run from project root:
#      python dataloader/dataset_eda.py
#
#  Output:
#      Saved report → output/dataset_eda_report.png
# =============================================================

import sys
from pathlib import Path
import os

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import CFG, LC_CLASSES, LC_COLOR_TO_CLASS
from dataloader.dataloader import build_dataframe

def analyze_dataset(df: pd.DataFrame, max_samples: int = 100):
    """
    Scans a representative subset of the dataset to calculate color, contrast,
    and class distribution metrics.
    """
    print(f"\n[EDA] Scanning up to {max_samples} samples from the dataset...")
    
    # Sample if the dataset is larger than max_samples
    if len(df) > max_samples:
        df_sampled = df.sample(n=max_samples, random_state=CFG["SEED"]).reset_index(drop=True)
    else:
        df_sampled = df.copy()

    # Accumulators for histograms
    r_hist = np.zeros(256, dtype=np.float64)
    g_hist = np.zeros(256, dtype=np.float64)
    b_hist = np.zeros(256, dtype=np.float64)
    gray_hist = np.zeros(256, dtype=np.float64)
    
    # Class pixel counters
    class_counts = {idx: 0 for idx in LC_CLASSES.keys()}
    total_pixels = 0

    # Image list with computed RMS contrast for sample sorting
    image_contrast_records = []

    for idx, row in enumerate(tqdm(df_sampled.itertuples(), total=len(df_sampled), desc="Processing images")):
        # 1. Load image (RGB)
        img_bgr = cv2.imread(row.image_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = img_rgb.shape
        
        # Calculate color histograms
        for i, hist_acc in enumerate([r_hist, g_hist, b_hist]):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            hist_acc += hist.flatten()

        # 2. Convert to grayscale for contrast analysis
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g_hist_curr = cv2.calcHist([gray], [0], None, [256], [0, 256])
        gray_hist += g_hist_curr.flatten()

        # Compute RMS contrast: std dev of normalized intensity [0, 1]
        gray_norm = gray.astype(np.float32) / 255.0
        rms_contrast = float(np.std(gray_norm))
        mean_intensity = float(np.mean(gray_norm))

        image_contrast_records.append({
            "image_path": row.image_path,
            "mask_path": row.mask_path,
            "rms_contrast": rms_contrast,
            "mean_intensity": mean_intensity,
            "img_rgb": img_rgb
        })

        # 3. Load mask for class distribution (for land_cover)
        if CFG["TASK"] == "land_cover":
            mask_bgr = cv2.imread(row.mask_path)
            if mask_bgr is not None:
                mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
                
                # Count class distribution efficiently
                for color, cls_idx in LC_COLOR_TO_CLASS.items():
                    match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
                    count = np.sum(match)
                    class_counts[cls_idx] += count
                    total_pixels += count

    # Normalize histograms
    total_images_processed = len(image_contrast_records)
    if total_images_processed > 0:
        r_hist /= total_images_processed
        g_hist /= total_images_processed
        b_hist /= total_images_processed
        gray_hist /= total_images_processed

    # Class percentages
    class_pcts = {}
    if total_pixels > 0:
        for cls_idx, count in class_counts.items():
            class_pcts[cls_idx] = (count / total_pixels) * 100
    else:
        # Fallback if no masks or different task
        class_pcts = {cls_idx: 0.0 for cls_idx in LC_CLASSES.keys()}

    # Sort images by contrast to select samples
    image_contrast_records = sorted(image_contrast_records, key=lambda x: x["rms_contrast"])
    
    # Pick lowest, median, and highest contrast images for visual demonstration
    low_contrast_item = image_contrast_records[0]
    med_contrast_item = image_contrast_records[len(image_contrast_records) // 2]
    high_contrast_item = image_contrast_records[-1]

    return {
        "r_hist": r_hist,
        "g_hist": g_hist,
        "b_hist": b_hist,
        "gray_hist": gray_hist,
        "class_pcts": class_pcts,
        "contrast_records": image_contrast_records,
        "samples": {
            "low": low_contrast_item,
            "median": med_contrast_item,
            "high": high_contrast_item
        }
    }

def make_report(results, output_path: Path):
    """
    Renders a stunning 2x2 grid report of color, contrast, and class distributions.
    """
    DARK_BG      = "#0a0a14"
    PANEL_BG     = "#121225"
    ACCENT_CYAN  = "#00d4ff"
    TEXT_WHITE   = "#f0f0f5"
    TEXT_MUTED   = "#8a8aa3"
    BORDER_COLOR = "#1f1f3d"

    # Setup figure
    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    fig.suptitle(
        f"DeepGlobe Satellite Dataset — Exploratory Data Analysis (EDA)\nSOICT Group 24 | Supervisor: Dr. Tran Nguyen Ngoc",
        fontsize=18, fontweight="bold", color=ACCENT_CYAN, y=0.96
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.2, left=0.06, right=0.94, top=0.88, bottom=0.08)

    # 1. Panel: Color Distribution (RGB Histograms)
    ax_color = fig.add_subplot(gs[0, 0])
    ax_color.set_facecolor(PANEL_BG)
    
    bins = np.arange(256)
    ax_color.fill_between(bins, results["r_hist"], color="#ff4d4d", alpha=0.3, label="Red Channel")
    ax_color.plot(bins, results["r_hist"], color="#ff1a1a", linewidth=1.5)
    
    ax_color.fill_between(bins, results["g_hist"], color="#4dff4d", alpha=0.3, label="Green Channel")
    ax_color.plot(bins, results["g_hist"], color="#1aff1a", linewidth=1.5)
    
    ax_color.fill_between(bins, results["b_hist"], color="#4d4dff", alpha=0.3, label="Blue Channel")
    ax_color.plot(bins, results["b_hist"], color="#1a1aff", linewidth=1.5)
    
    ax_color.set_title("Color Channel Pixel Intensity Distribution", color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=10)
    ax_color.set_xlabel("Pixel Value (0-255)", color=TEXT_MUTED)
    ax_color.set_ylabel("Average Pixel Count per Image", color=TEXT_MUTED)
    ax_color.tick_params(colors=TEXT_MUTED)
    ax_color.grid(True, color=BORDER_COLOR, linestyle="--", alpha=0.5)
    ax_color.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_WHITE)

    # 2. Panel: Contrast Level Analysis
    ax_contrast = fig.add_subplot(gs[0, 1])
    ax_contrast.set_facecolor(PANEL_BG)
    
    # Plot grayscale intensity distribution
    ax_contrast.fill_between(bins, results["gray_hist"], color="#cccccc", alpha=0.25, label="Grayscale Intensity")
    ax_contrast.plot(bins, results["gray_hist"], color="#ffffff", linewidth=2)
    
    # Calculate dataset-wide contrast metrics
    rms_values = [r["rms_contrast"] for r in results["contrast_records"]]
    mean_rms = np.mean(rms_values)
    std_rms = np.std(rms_values)
    
    # High vs Low contrast evaluation threshold (0.15 is typical threshold for satellite)
    # Highlight threshold region
    ax_contrast.axvline(x=128, color="#ffaa00", linestyle=":", linewidth=1.5, label="Midpoint (128)")
    
    # Contrast Text box
    stats_text = (
        f"Contrast Level Stats:\n"
        f"  • Avg RMS Contrast : {mean_rms:.3f}\n"
        f"  • Contrast Std Dev  : {std_rms:.3f}\n"
        f"  • Min RMS Contrast  : {min(rms_values):.3f}\n"
        f"  • Max RMS Contrast  : {max(rms_values):.3f}\n\n"
        f"Overall evaluation:\n"
        f"  • High Contrast (>0.18) : {sum(1 for v in rms_values if v > 0.18)} imgs\n"
        f"  • Medium Contrast (0.12-0.18) : {sum(1 for v in rms_values if 0.12 <= v <= 0.18)} imgs\n"
        f"  • Low Contrast (<0.12)  : {sum(1 for v in rms_values if v < 0.12)} imgs"
    )
    ax_contrast.text(
        0.05, 0.45, stats_text, transform=ax_contrast.transAxes,
        fontsize=10.5, color=TEXT_WHITE, family="monospace",
        bbox=dict(facecolor=DARK_BG, edgecolor=ACCENT_CYAN, boxstyle="round,pad=0.8", alpha=0.8)
    )

    ax_contrast.set_title("Contrast & Grayscale Intensity Profile", color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=10)
    ax_contrast.set_xlabel("Pixel Value (0-255)", color=TEXT_MUTED)
    ax_contrast.set_ylabel("Average Pixel Count per Image", color=TEXT_MUTED)
    ax_contrast.tick_params(colors=TEXT_MUTED)
    ax_contrast.grid(True, color=BORDER_COLOR, linestyle="--", alpha=0.5)
    ax_contrast.legend(facecolor=PANEL_BG, edgecolor=BORDER_COLOR, labelcolor=TEXT_WHITE, loc="upper right")

    # 3. Panel: Class Distribution (Land Cover)
    ax_class = fig.add_subplot(gs[1, 0])
    ax_class.set_facecolor(PANEL_BG)
    
    class_names = [v[0] for v in LC_CLASSES.values()]
    class_colors = [np.array(v[1]) / 255.0 for v in LC_CLASSES.values()]
    pct_values = [results["class_pcts"][idx] for idx in LC_CLASSES.keys()]
    
    bars = ax_class.barh(class_names, pct_values, color=class_colors, edgecolor="#ffffff", linewidth=0.5)
    ax_class.invert_yaxis()  # top-down list
    
    # Add labels on bars
    for bar in bars:
        width = bar.get_width()
        ax_class.text(
            width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
            color=TEXT_WHITE, va="center", ha="left", fontsize=9.5, fontweight="bold"
        )
        
    ax_class.set_title("Land Cover Class Pixel Distribution (Dataset Balance)", color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=10)
    ax_class.set_xlabel("Pixel Percentage (%)", color=TEXT_MUTED)
    ax_class.tick_params(colors=TEXT_MUTED)
    ax_class.set_xlim(0, max(pct_values) + 8)  # Leave room for text labels
    ax_class.grid(True, axis="x", color=BORDER_COLOR, linestyle="--", alpha=0.5)

    # 4. Panel: Sample contrast comparison
    ax_grid = fig.add_subplot(gs[1, 1])
    ax_grid.set_facecolor(DARK_BG)
    ax_grid.axis("off")
    ax_grid.set_title("Sample Satellite Images (Contrast Variations)", color=TEXT_WHITE, fontsize=13, fontweight="bold", pad=15)

    # Divide Panel 4 into 3 sub-images using GridSpec inside it
    sub_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1], wspace=0.15)
    
    samples_info = [
        ("Low Contrast", results["samples"]["low"]),
        ("Median Contrast", results["samples"]["median"]),
        ("High Contrast", results["samples"]["high"])
    ]
    
    for sub_idx, (label, record) in enumerate(samples_info):
        sub_ax = fig.add_subplot(sub_gs[0, sub_idx])
        sub_ax.imshow(cv2.resize(record["img_rgb"], (256, 256)))
        sub_ax.axis("off")
        
        # Overlay standard deviation details
        sub_ax.set_title(
            f"{label}\nRMS: {record['rms_contrast']:.3f}", 
            color=TEXT_WHITE, fontsize=10, pad=8
        )
        
        # Border around the sample image
        for spine in sub_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(ACCENT_CYAN if sub_idx == 1 else BORDER_COLOR)
            spine.set_linewidth(1.5)

    # Draw border lines on all panels
    for ax in [ax_color, ax_contrast, ax_class]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(BORDER_COLOR)
            spine.set_linewidth(1)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n[EDA] Saved visual report → {output_path}")
    plt.close()

def main():
    print("=" * 65)
    print("  dataset_eda.py — Dataset Exploratory Data Analysis")
    print("=" * 65)
    
    # Find dataset
    df = build_dataframe(CFG["DATA_ROOT"])
    if len(df) == 0:
        print("[ERROR] No image-mask pairs found. Check DATA_ROOT in config.py!")
        return

    # Run analysis
    results = analyze_dataset(df, max_samples=100)
    
    # Make visual report
    output_dir = Path(__file__).resolve().parent / "images"
    output_path = output_dir / "dataset_eda_report.png"
    make_report(results, output_path)
    
    print("\n[EDA] Diagnostics and metrics analysis completed successfully.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
