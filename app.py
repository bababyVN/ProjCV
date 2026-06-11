# =============================================================
#  app.py — Gradio Satellite Image Segmentation Demo
#
#  Allows users to upload a satellite image, select SwinFAN-v1 or
#  SwinFAN-v3 models, run inference, and instantly adjust mask
#  overlay opacity in real-time without re-predicting.
#
#  Optimised for "potatoes" (CPU-only, lazy model loading,
#  single-pass Potato Mode, and instant UI start).
#
#  Run with: python app.py
# =============================================================

import sys
from pathlib import Path
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
import gc
import time
import copy

# Resolve project path imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Monkeypatch gradio_client to prevent a TypeError when parsing JSON schemas with boolean additionalProperties (common in Gradio 5 with gr.Image)
try:
    import gradio_client.utils as client_utils
    
    # Outer function patch
    _orig_json_schema_to_python_type_outer = client_utils.json_schema_to_python_type
    def _patched_json_schema_to_python_type_outer(schema):
        try:
            return _orig_json_schema_to_python_type_outer(schema)
        except TypeError as e:
            if "bool" in str(e) or "iterable" in str(e):
                return "Any"
            raise e
    client_utils.json_schema_to_python_type = _patched_json_schema_to_python_type_outer

    # Inner function patch
    _orig_json_schema_to_python_type = client_utils._json_schema_to_python_type
    def _patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "bool"
        return _orig_json_schema_to_python_type(schema, defs)
    client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception as e:
    import traceback
    print(f"[app] Monkeypatch failed: {e}")
    traceback.print_exc()

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from config import CFG, LC_CLASSES
from dataloader.dataloader import (
    rgb_mask_to_index,
    index_to_rgb,
    get_val_transform
)
from model.models import build_model


# ─────────────────────────────────────────────────────────────
# Model Cache & Lazy Loader
# ─────────────────────────────────────────────────────────────
LOADED_MODELS = {
    "v1": None,
    "v3": None
}

def load_and_cache_model(version: str) -> torch.nn.Module:
    """
    Load the selected model version and unload the other version to save RAM.
    """
    global LOADED_MODELS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    other_version = "v3" if version == "v1" else "v1"
    
    # ── 1. Unload the other model to free RAM/VRAM ─────────────────────────
    if LOADED_MODELS[other_version] is not None:
        print(f"[app] Unloading SwinFAN-{other_version} to save RAM...")
        LOADED_MODELS[other_version] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # ── 2. Check if selected model is already loaded ──────────────────────
    if LOADED_MODELS[version] is not None:
        print(f"[app] SwinFAN-{version} already loaded.")
        return LOADED_MODELS[version]
        
    # ── 3. Build and load the weights ─────────────────────────────────────
    print(f"[app] Building SwinFAN-{version} architecture...")
    if version == "v1":
        cfg = copy.deepcopy(CFG)
        cfg["ARCH"] = "swinfan"
        cfg["NUM_CLASSES"] = 7
        cfg["DEVICE"] = device
        
        ckpt_path = Path("output/best_model.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SwinFAN-v1 checkpoint not found at {ckpt_path.resolve()}")
            
        model = build_model(cfg).to(device)
        print(f"[app] Loading SwinFAN-v1 checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
    elif version == "v3":
        # Monkeypatch SwinV2BaseEncoder to bypass the 503MB SatlasPretrain weights download.
        from encoder.swin_encoder import SwinV2BaseEncoder
        SwinV2BaseEncoder._load_satlas_weights = lambda self, backbone: print(
            "[app] Skipping Satlas Pretrain download since we load best_model_v3.pth immediately after"
        )
        
        cfg = copy.deepcopy(CFG)
        cfg["ARCH"] = "swinfan_v3"
        cfg["NUM_CLASSES"] = 7
        cfg["DEVICE"] = device
        
        ckpt_path = Path("output/best_model_v3.pth")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SwinFAN-v3 checkpoint not found at {ckpt_path.resolve()}")
            
        model = build_model(cfg).to(device)
        print(f"[app] Loading SwinFAN-v3 checkpoint: {ckpt_path.name}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        
    else:
        raise ValueError(f"Unknown version: {version}")
        
    LOADED_MODELS[version] = model
    return model


# ─────────────────────────────────────────────────────────────
# Normalisation Utility for single patches
# ─────────────────────────────────────────────────────────────
_TRANSFORM = get_val_transform()

def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    """Normalise image using ImageNet constants and return (1, 3, H, W) tensor."""
    transformed = _TRANSFORM(image=image_rgb)["image"]
    return transformed.unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Potato Mode (Fast Single-Pass Inference)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def potato_inference(model: torch.nn.Module, image_rgb: np.ndarray) -> np.ndarray:
    """
    Run inference in a single forward pass by resizing the image to 512×512.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = image_rgb.shape[:2]
    
    resized_img = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_AREA)
    tensor = preprocess_image(resized_img).to(device)
    
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    pred_indices = probs.squeeze(0).argmax(dim=0).cpu().numpy().astype(np.uint8)
    
    upscaled_indices = cv2.resize(pred_indices, (W, H), interpolation=cv2.INTER_NEAREST)
    return upscaled_indices


# ─────────────────────────────────────────────────────────────
# Standard Mode (Full-Res sliding window)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sliding_window_inference_gradio(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    patch_size: int = 512,
    overlap: int = 64,
    num_classes: int = 7,
    progress=gr.Progress()
) -> np.ndarray:
    """
    Detailed sliding window inference. Yields progress updates to Gradio.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = image_rgb.shape[:2]
    
    logit_map = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)
    
    gy = np.hanning(patch_size).reshape(-1, 1)
    gx = np.hanning(patch_size).reshape(1, -1)
    gauss_kernel = (gy * gx).astype(np.float32)
    
    step = patch_size - overlap
    ys = list(range(0, H - patch_size, step)) + [H - patch_size]
    xs = list(range(0, W - patch_size, step)) + [W - patch_size]
    ys = sorted(set(max(0, y) for y in ys))
    xs = sorted(set(max(0, x) for x in xs))
    
    total_patches = len(ys) * len(xs)
    patch_count = 0
    
    progress(0.0, desc="Starting sliding window inference...")
    
    for y in ys:
        for x in xs:
            patch = image_rgb[y : y + patch_size, x : x + patch_size]
            tensor = preprocess_image(patch).to(device)
            
            logits = model(tensor)
            
            if logits.shape[-2:] != (patch_size, patch_size):
                logits = F.interpolate(logits, size=(patch_size, patch_size),
                                       mode="bilinear", align_corners=False)
                                       
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            logit_map[:, y : y + patch_size, x : x + patch_size] += probs * gauss_kernel[np.newaxis, ...]
            weight_map[y : y + patch_size, x : x + patch_size] += gauss_kernel
            
            patch_count += 1
            progress(patch_count / total_patches, desc=f"Processing patch {patch_count}/{total_patches}...")
            
    weight_map = np.maximum(weight_map, 1e-8)
    logit_map /= weight_map[np.newaxis, ...]
    
    pred_indices = logit_map.argmax(axis=0).astype(np.uint8)
    return pred_indices


# ─────────────────────────────────────────────────────────────
# Callback Functions
# ─────────────────────────────────────────────────────────────
def run_segmentation(
    input_img,
    model_ver,
    infer_mode,
    alpha,
    progress=gr.Progress()
):
    if input_img is None:
        return None, None, "### ⚠️ Error\nPlease upload an image first.", None
        
    start_time = time.time()
    
    try:
        # 1. Resolve model name
        version_key = "v1" if "v1" in model_ver else "v3"
        
        # 2. Get/load model
        model = load_and_cache_model(version_key)
        
        # 3. Perform Inference
        if infer_mode == "Potato Mode (Fast Single-Pass)":
            pred_indices = potato_inference(model, input_img)
        else:
            pred_indices = sliding_window_inference_gradio(
                model, input_img, patch_size=512, overlap=64, num_classes=7, progress=progress
            )
            
        elapsed_time = time.time() - start_time
        
        # 4. Generate visual outputs
        pred_rgb = index_to_rgb(pred_indices)
        blended = cv2.addWeighted(input_img, 1.0 - alpha, pred_rgb, alpha, 0)
        
        # 5. Compute Class Percentages
        total_pixels = pred_indices.size
        percentages = {}
        for cls_idx, (cls_name, _) in LC_CLASSES.items():
            count = np.sum(pred_indices == cls_idx)
            percentages[cls_name] = (count / total_pixels) * 100
            
        # 6. Generate markdown report
        pct_rows = [f"| **{name}** | {pct:.2f}% |" for name, pct in percentages.items()]
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        
        report_md = f"""
## Inference Report
- **Model Used**: {'Swin-B 100M, Satlas' if version_key=='v3' else 'Swin-T 35M, ImageNet'}
- **Inference Mode**: {infer_mode}
- **Device**: {device_name}
- **Time Elapsed**: **{elapsed_time:.3f} seconds**

### Land Cover Area Distribution
| Class | Area Coverage |
| :--- | :--- |
{chr(10).join(pct_rows)}
"""
        # Return: Predicted mask, blended overlay, report, and pred_rgb state
        return pred_rgb, blended, report_md, pred_rgb
        
    except Exception as e:
        import traceback
        err_msg = f"### ❌ Inference Failed\nAn error occurred during inference: `{str(e)}`\n\n```python\n{traceback.format_exc()}\n```"
        print(err_msg)
        return None, None, err_msg, None


def update_overlay_only(input_img, pred_rgb_state, alpha):
    """
    Perform instant blending of input image and cached predicted mask when alpha changes.
    Does not run the model, executing in less than a millisecond.
    """
    if input_img is None or pred_rgb_state is None:
        return None
    blended = cv2.addWeighted(input_img, 1.0 - alpha, pred_rgb_state, alpha, 0)
    return blended


# ─────────────────────────────────────────────────────────────
# UI Page Construction
# ─────────────────────────────────────────────────────────────
# Premium dark-theme custom CSS overriding default colors to force dark mode
custom_css = """
/* Force dark background for the whole page and Gradio container */
body, html, .gradio-container {
    background-color: #030712 !important;
    color: #f3f4f6 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Style cards and panels with clean dark color and fine borders */
.block {
    background-color: #0f172a !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
}

/* Style headings */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Style labels inside components */
.block span.label {
    color: #94a3b8 !important;
}

/* Style main CTA buttons */
button.primary {
    background: linear-gradient(135deg, #0284c7, #4f46e5) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(2, 132, 199, 0.4) !important;
}

/* Style sliders and inputs */
.gr-slider-input, input[type="range"] {
    accent-color: #38bdf8 !important;
}

/* Style header title card */
.title-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(3, 7, 18, 0.95)) !important;
    border-radius: 16px !important;
    padding: 28px !important;
    margin-bottom: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
    text-align: center;
}
.title-card h1 {
    font-size: 2.8em !important;
    font-weight: 800 !important;
    margin: 0 0 12px 0 !important;
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}
.title-card p {
    color: #94a3b8 !important;
    font-size: 1.15em !important;
    margin: 0 !important;
}

/* Style the class legend */
.legend-card {
    background: rgba(15, 23, 42, 0.7) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.legend-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)) !important;
    gap: 12px !important;
}
.legend-item {
    display: flex !important;
    align-items: center !important;
    font-size: 0.95em !important;
    color: #e2e8f0 !important;
}
.legend-color-box {
    width: 22px !important;
    height: 22px !important;
    border-radius: 6px !important;
    margin-right: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
}
"""

# JavaScript snippet executed on load to force the browser into Gradio's native dark mode
force_dark_js = """
() => {
    document.body.classList.add('dark');
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="sky", secondary_hue="indigo", neutral_hue="slate"), css=custom_css, js=force_dark_js) as demo:
    
    # State variable to cache predicted mask RGB array for real-time opacity changes
    predicted_mask_state = gr.State(None)
    
    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(
        """
        <div class="title-card">
            <h1>SwinFAN Satellite Segmentation Dashboard</h1>
            <p>Load SwinFAN Models, segment remote sensing imagery, and blend output masks in real-time.</p>
        </div>
        """
    )
    
    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items_html = ""
    for idx, (cls_name, color) in LC_CLASSES.items():
        color_rgb = f"rgb({color[0]}, {color[1]}, {color[2]})"
        legend_items_html += f"""
        <div class="legend-item">
            <div class="legend-color-box" style="background-color: {color_rgb};"></div>
            <span>{cls_name}</span>
        </div>
        """
    gr.HTML(
        f"""
        <div class="legend-card">
            <h3 style="margin: 0 0 10px 0; color: #38bdf8;">Land Cover Classes</h3>
            <br>
            <div class="legend-grid">
                {legend_items_html}
            </div>
        </div>
        """
    )
    
    with gr.Row():
        # ── Controls Left Panel ────────────────────────────────────────────────
        with gr.Column(scale=1):
            
            # Upload satellite image uploader

            gr.Markdown("### Upload Satellite Image")
            input_image_disp = gr.Image(type="numpy", label="Input Satellite Image")

            # Predict Button
            predict_btn = gr.Button(" Run Segmentation", variant="primary")
            gr.Markdown("### Configuration & Input")
            
            # Model Selection
            model_ver_input = gr.Radio(
                choices=["SwinFAN-T (ImageNet1K Swin-T Backbone, 35M params)", "SwinFAN-B (Satlas Swin-B Backbone, 100M params)"],
                value="SwinFAN-B (Satlas Swin-B Backbone, 100M params)",
                label="Model Version"
            )
            
            # Execution Mode
            infer_mode_input = gr.Radio(
                choices=["Potato Mode (Fast Single-Pass)", "Standard (Full-Res sliding window)"],
                value="Standard (Full-Res sliding window)",
                label="Inference Execution Mode",
                info=" Potato Mode downsamples to 512px and runs instantly, while Standard use sliding windows."
            )
            
        # ── Main Visual Panels Middle/Right ───────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Visual Dashboard")
            
            with gr.Row():
                pred_mask_disp = gr.Image(type="numpy", label="Predicted Mask")
                overlay_disp = gr.Image(type="numpy", label="Overlay Blend")

            # Opacity slider
            alpha_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.2,
                step=0.01,
                label="Mask Visual Overlay Opacity"
            )
                
            gr.Markdown("---")
            # Metrics Section
            report_disp = gr.Markdown("### Performance & Metrics\n*Run segmentation to view stats...*")

    # ── Wire Up Event Listeners ───────────────────────────────────────────────

    # 1. Trigger segmentation inference and save predicted mask to state
    predict_btn.click(
        fn=run_segmentation,
        inputs=[
            input_image_disp,
            model_ver_input,
            infer_mode_input,
            alpha_input
        ],
        outputs=[
            pred_mask_disp,
            overlay_disp,
            report_disp,
            predicted_mask_state
        ]
    )

    # 2. When the alpha slider value changes, update the overlay instantly in real-time without running the model
    alpha_input.change(
        fn=update_overlay_only,
        inputs=[
            input_image_disp,
            predicted_mask_state,
            alpha_input
        ],
        outputs=[
            overlay_disp
        ]
    )

# ── Launch Gradio ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1", share=False)
