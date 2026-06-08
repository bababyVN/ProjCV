# =============================================================
#  download_kaggle_dataset.py — Download DeepGlobe from Kaggle
#
#  Uses kagglehub (no manual kaggle CLI setup needed).
#  Copies train/, valid/, test/ folders into the project's data/ directory.
#
#  Run with: python download_kaggle_dataset.py
# =============================================================
import shutil
from pathlib import Path
import kagglehub
# ── Where to put the data in this project ────────────────────
DEST = Path(__file__).parent / "data"
# ── Splits to copy ───────────────────────────────────────────
SPLITS = ["train", "valid", "test"]
# ── Download via kagglehub ────────────────────────────────────
print("Downloading DeepGlobe Land Cover dataset from Kaggle...")
print("(This may take a while — dataset is ~3.4 GB)\n")
path = kagglehub.dataset_download("balraj98/deepglobe-land-cover-classification-dataset")
src  = Path(path)
print(f"\nKagglehub cache path: {src}")
# ── Show what was downloaded ──────────────────────────────────
print("\nContents of downloaded path:")
for f in sorted(src.iterdir()):
    label = "[DIR]" if f.is_dir() else "[FILE]"
    print(f"  {label} {f.name}")
# ── Copy each split ───────────────────────────────────────────
for split in SPLITS:
    split_src  = src / split
    split_dest = DEST / split
    if not split_src.exists():
        print(f"\n[SKIP] '{split}' folder not found in download — skipping.")
        continue
    split_dest.mkdir(parents=True, exist_ok=True)
    files = list(split_src.glob("*"))
    print(f"\nCopying {split}/ ({len(files)} files) -> {split_dest} ...")
    for i, f in enumerate(files, 1):
        dest_f = split_dest / f.name
        if not dest_f.exists():
            shutil.copy2(f, dest_f)
        if i % 200 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] done")
    # Verify
    sat_count  = len(list(split_dest.glob("*_sat.jpg")))
    mask_count = len(list(split_dest.glob("*_mask.png")))
    print(f"  Verified — *_sat.jpg: {sat_count}  |  *_mask.png: {mask_count}")
print(f"\nAll done! Dataset ready at: {DEST}")
print("You can now run: python -m dataloader.dataset")