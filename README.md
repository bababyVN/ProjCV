# DeepGlobe Semantic Segmentation: SwinFAN & Hybrid Models

This repository contains a PyTorch implementation of semantic segmentation models for earth observation tasks based on the DeepGlobe dataset. It features the **SwinFAN** architecture (Swin Transformer Encoder + Attention-Guided Decoder) and a legacy **Hybrid CNN-Transformer** model.

It supports two main tasks:
1. **Land Cover Classification**: 7-class multi-label semantic segmentation (DeepGlobe Land Cover).
2. **Road Extraction**: Binary semantic segmentation (DeepGlobe Road Extraction).

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
│   ├── swin_encoder.py       # Swin-T backbone pre-trained on ImageNet, combined with a lightweight CNN stem
│   ├── custom_encoder.py     # Legacy hybrid CNN (ResNet-34) + self-attention transformer encoder
│   └── transformer_block.py  # Standard Transformer blocks used in the legacy CNN-Transformer encoder
│
├── decoder/                  # Feature upsampling and cross-attention fusion modules
│   ├── swinfan_decoder.py    # SwinFAN decoder incorporating Attention Gates (AG) for multi-scale skip connections
│   └── unet_decoder.py       # Standard UNet decoder with convolutional upsampling blocks
│
├── model/                    # Assembly of final segmentation models
│   └── models.py             # SwinFANModel, HybridSegModel definitions, and build_model() factory
│
├── helper/                   # Loss functions and evaluation metrics
│   ├── losses.py             # HybridLoss (combining Focal Loss and Dice Loss with task weightings)
│   └── metrics.py            # mIoU, Overall Pixel Accuracy, and F1-Score calculations
│
├── scripts/                  # Entrypoint scripts for execution and training
│   ├── download_kaggle_dataset.py # Automated helper to fetch datasets from Kaggle using kagglehub
│   ├── train.py              # Main training script (implements train loop, validation, AMP, and saving checkpoints)
│   ├── kaggle.ipynb          # Jupyter notebook configuration for training in Kaggle environments
│   └── model_architecture.png # Reference diagram of the network architecture
│
└── output/                   # Directory where trained models (.pth) and logs are written
```

---

## Key Architectural Details

### 1. SwinFAN (`ARCH: "swinfan"`)
- **Encoder (`encoder/swin_encoder.py`)**: Uses a pre-trained Swin Transformer (`swin_t`) as a backbone to learn hierarchical context with shifted-window self-attention. To keep fine details from high-resolution layers, we augment it with a lightweight CNN stem (outputs features at strides 1 and 2).
- **Decoder (`decoder/swinfan_decoder.py`)**: Integrates **Attention Gates (AG)** at each decoding level to filter features forwarded from the encoder skips. This focuses the model's capacity on relevant objects, improving boundaries and edge localization.
- **Model (`model/models.py`)**: Packages the SwinEncoder, SwinFANDecoder, and a 1x1 Convolutional Segmentation Head.

### 2. Hybrid Model (`ARCH: "hybrid"`)
- **Encoder (`encoder/custom_encoder.py`)**: Combines a standard CNN encoder (ResNet-34) for local feature representation with axial-attention Transformer blocks in the bottleneck to capture global dependencies.
- **Decoder (`decoder/unet_decoder.py`)**: A traditional UNet-style decoder using standard skip connections.

---

## Configurable Hyperparameters (`config.py`)

All hyperparameters, task settings, and classes are configured in a centralized dictionary `CFG` within `config.py`:
- `TASK`: Switch between `"land_cover"` (7 classes) or `"road"` (binary classification, 1 class).
- `ARCH`: Choose model architecture (`"swinfan"` or `"hybrid"`).
- `DATA_ROOT` / `OUTPUT_DIR`: Path setups for local data or Kaggle environment directories.
- `IMG_SIZE`: Target height/width of training patches (e.g., 512).
- `EPOCHS` / `BATCH_SIZE` / `LR` / `WEIGHT_DECAY`: Standard optimizer and scheduling hyperparameters.
- `FOCAL_WEIGHT` / `DICE_WEIGHT`: Balance parameters between Focal Loss and Dice Loss.

---

## Getting Started

### 1. Install Dependencies
Make sure you have Python installed, then set up the packages:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
The dataset can be fetched automatically from Kaggle. Run the following helper:
```bash
python scripts/download_kaggle_dataset.py
```
This downloads and organizes files under the `data/` folder.

### 3. Run Shape Tests
Verify that encoders, decoders, and data ingestion are set up properly on your system:
```bash
# Test the dataloader loading logic
python -m dataloader.dataloader

# Test SwinEncoder output shapes
python encoder/swin_encoder.py

# Test end-to-end model construction and shape pass-through
python model/models.py
```

### 4. Run Model Training
Start the model training and evaluation using:
```bash
python scripts/train.py
```
This script automatically:
- Fixes random seeds for reproducibility.
- Applies data augmentations (flips, rotates, affine crops, color jittering) via `albumentations`.
- Implements Automatic Mixed Precision (AMP) to speed up execution.
- Evaluates metrics (**Mean IoU**, **Pixel Accuracy**, and **F1-Score**) at each epoch.
- Saves the best checkpoint to the `output/` folder based on validation Mean IoU.
