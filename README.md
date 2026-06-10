# DeepGlobe Semantic Segmentation: SwinFAN Hybrid Models

This repository contains a PyTorch implementation of semantic segmentation models for earth observation tasks based on the DeepGlobe dataset. It features the **SwinFAN** architecture (Swin Transformer Encoder + Attention-Guided Decoder).

---

## Model Versions Overview

We support two primary versions of the SwinFAN architecture, optimized for progressively higher accuracy and better utilization of remote sensing pretraining:

| Feature / Version | SwinFAN-v1 (Baseline) | SwinFAN-v3 (Satlas RS Pretrained)* |
| :--- | :--- | :--- |
| **Backbone** | Swin-Tiny (`swin_t`) | Swin-v2-Base (`swin_v2_b`) |
| **Pretraining** | ImageNet-1K | SatlasPretrain Aerial (`Aerial_SwinB_SI`) |
| **Model Size** | ~35M parameters | ~100M parameters |
| **Loss Function** | Focal (50%) + Dice (50%) | Focal (30%) + Dice (30%) + Lovász (40%) |
| **Batch Size** | 8 | 4 (Grad Accum = 2, Effective = 8) |
| **Epochs / LR** | 30 Epochs, LR: 6e-4 | 50 Epochs, LR: 3e-4 |
| **Target mIoU (6-Class)** | ~0.52 | **~0.65 – 0.72** |
| **Training Script** | `scripts/train.py` | `scripts/train_v3.py` |
| **Saved Checkpoint** | `output/best_model.pth` | `output/best_model_v3.pth` |

*\*SwinFAN-v3 builds upon the architecture improvements of SwinFAN-v2 (incorporating a wider backbone, multi-component hybrid loss, and gradient accumulation optimization).*

---

## Project Structure

Below is an overview of the directory structure and the roles of the key files:

```
ProjCv/
├── config.py                 # Global hyperparameters, class details, paths, and task switches
├── requirements.txt          # Package dependencies (PyTorch, Albumentations, OpenCV, etc.)
├── LICENSE                   # Project license (MIT)
├── .gitignore                # Git ignore configurations
│
├── data/                     # Root folder for datasets (split into train, valid, test)
│   ├── train/                # Raw training satellite images (*_sat.jpg) and label masks (*_mask.png)
│   ├── valid/                # Validation images and masks
│   └── test/                 # Test images
│
├── dataloader/               # Dataset classes, loading utilities, and data augmentation
│   ├── dataloader.py         # DeepGlobe Dataset class, fast online patch sampling, and pipelines
│   ├── check_dataloader.py   # Utility script to verify dataset loading and shape correctness
│   ├── dataset_eda.py        # Exploratory Data Analysis (EDA) on labels, sizes, and colors
│   ├── preprocessing_showcase.py # Visualizations of patch-extraction and image augmentations
│   └── infer.py              # Script for full-image inference using sliding-window prediction
│
├── encoder/                  # Multi-scale feature extractor backbones
│   ├── swin_encoder.py       # Backbones for v1 (Swin-T), v2 (Swin-B), and v3 (Satlas Swin-v2-B)
│   ├── custom_encoder.py     # Legacy hybrid CNN (ResNet-34) + self-attention transformer encoder
│   └── transformer_block.py  # Standard Transformer blocks used in the legacy CNN-Transformer encoder
│
├── decoder/                  # Feature upsampling and cross-attention fusion modules
│   ├── swinfan_decoder.py    # SwinFAN decoder incorporating Attention Gates (AG) for multi-scale skip connections
│   └── unet_decoder.py       # Standard UNet decoder with convolutional upsampling blocks
│
├── model/                    # Assembly of final segmentation models
│   └── models.py             # SwinFANModel (for v1, v2, v3), HybridSegModel definitions, and build_model() factory
│
├── helper/                   # Loss functions and evaluation metrics
│   ├── losses.py             # Focal + Dice Hybrid Loss and Lovász Softmax Loss functions
│   └── metrics.py            # mIoU, Overall Pixel Accuracy, and F1-Score calculations
│
├── scripts/                  # Entrypoint scripts for execution and training
│   ├── download_kaggle_dataset.py # Automated helper to fetch datasets from Kaggle using kagglehub
│   ├── train.py              # Training script for SwinFAN-v1
│   ├── train_v3.py           # Training script for SwinFAN-v3
│   ├── evaluate_model.py     # Detailed per-class evaluation script (works on all checkpoint versions)
│   ├── kaggle.ipynb          # Jupyter notebook for SwinFAN-v1 on Kaggle
│   ├── kagglev3.ipynb        # Jupyter notebook for SwinFAN-v3 on Kaggle
│   └── model_architecture.png # Reference diagram of the network architecture
│
└── output/                   # Directory where trained models (.pth) and logs are written
```

---

## Key Architectural Details

### 1. SwinFAN-v1 (`ARCH: "swinfan"`)
- **Encoder**: Uses a pre-trained Swin Transformer (`swin_t`) as a backbone to learn hierarchical context. Augmented with a lightweight CNN stem (stride-1 raw pixels and stride-2 convolution) to preserve high-resolution spatial features.
- **Decoder**: Integrates **Attention Gates (AG)** at each decoding level to filter skip-connection features, improving boundaries and edge localization.
- **Loss**: Balanced 50% Focal Loss and 50% Dice Loss.

### 2. SwinFAN-v3 (`ARCH: "swinfan_v3"`) — Based on v2 Architecture
- **Encoder**: Upgraded to the modern Swin-v2-Base (`swin_v2_b`) architecture. This features a wider backbone than v1, doubling the channel width from `[3, 48, 96, 192, 384, 768]` to `[3, 48, 128, 256, 512, 1024]`, totaling ~100M parameters.
- **Pretraining**: Initialized with weights from **SatlasPretrain** (`Aerial_SwinB_SI`), which was pretrained by Allen AI on 302 million remote sensing labels using 0.5–2 m/pixel aerial imagery. This domain-specific pretraining is a perfect scale match for DeepGlobe (50 cm/pixel) and yields a significant performance boost (+3–5% mIoU).
- **Loss**: Incorporates **Lovász Softmax Loss** (40%) along with Focal (30%) and Dice (30%) to directly optimize the Mean Intersection-over-Union (IoU) metric (inherited from v2).
- **Optimizations**: Implements **Gradient Accumulation** (accumulate 2 steps of batch size 4 to get effective batch size 8) to train the ~100M parameter model on a 16GB Tesla T4 GPU without Out-Of-Memory (OOM) errors (inherited from v2).
- **Auto Weight Caching**: On the first training run, the script automatically downloads the ~503 MB pre-trained checkpoint from Hugging Face and caches it in `/kaggle/working/weights/` (on Kaggle) or `~/.cache/satlas/` (locally).

---

## Getting Started & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
```bash
python scripts/download_kaggle_dataset.py
```

### 3. Running Model Training

You can run training for any version independently. Each version outputs to a separate checkpoint path:

- **Train SwinFAN-v1**:
  ```bash
  python scripts/train.py
  ```
- **Train SwinFAN-v3**:
  ```bash
  python scripts/train_v3.py
  ```

### 4. Detailed Evaluation

A dedicated evaluation script `scripts/evaluate_model.py` computes and displays a visual per-class evaluation table including IoU and F1-score for all classes, plus overall Pixel Accuracy and the standard 6-class vs 7-class metrics.

Run evaluation by passing the desired checkpoint path:
```bash
# Evaluate SwinFAN-v1
python scripts/evaluate_model.py output/best_model.pth

# Evaluate SwinFAN-v3
python scripts/evaluate_model.py output/best_model_v3.pth
```

### 5. Running on Kaggle

Two pre-configured Jupyter notebooks are available to easily run training on Kaggle GPUs:
- Use `scripts/kaggle.ipynb` for **SwinFAN-v1**.
- Use `scripts/kagglev3.ipynb` for **SwinFAN-v3**.

Both notebooks auto-detect the DeepGlobe dataset root directory and prepare a `.zip` file of your outputs upon completion.
