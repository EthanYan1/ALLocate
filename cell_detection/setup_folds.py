from pathlib import Path
import shutil
import random
import math


dataset_root = Path("data")
train_images = dataset_root / "train" / "images"
train_labels = dataset_root / "train" / "labels"
folds_root = dataset_root / "folds"

n_folds = 5
seed = 42

# valid image extensions
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

all_images = [p for p in train_images.iterdir() if p.suffix.lower() in image_exts]
all_images.sort()

pairs = []
missing_labels = []

for img_path in all_images:
    label_path = train_labels / f"{img_path.stem}.txt"
    if label_path.exists():
        pairs.append((img_path, label_path))
    else:
        missing_labels.append(img_path.name)

print(f"Found {len(all_images)} images total")
print(f"Found {len(pairs)} image-label pairs")
print(f"Missing labels for {len(missing_labels)} images")

if missing_labels:
    print("\nImages missing labels:")
    for name in missing_labels[:20]:
        print(" ", name)
    if len(missing_labels) > 20:
        print(f"  ... and {len(missing_labels) - 20} more")

if len(pairs) == 0:
    raise RuntimeError("No valid image-label pairs found.")

random.seed(seed)
random.shuffle(pairs)

fold_sizes = [len(pairs) // n_folds] * n_folds
for i in range(len(pairs) % n_folds):
    fold_sizes[i] += 1

folds = []
start = 0
for size in fold_sizes:
    folds.append(pairs[start:start + size])
    start += size

if folds_root.exists():
    print(f"\nWarning: {folds_root} already exists.")
    response = input("Delete and recreate it? [y/N]: ").strip().lower()
    if response == "y":
        shutil.rmtree(folds_root)
    else:
        raise RuntimeError("Aborted to avoid overwriting existing folds.")

for i in range(n_folds):
    (folds_root / f"fold{i+1}" / "images").mkdir(parents=True, exist_ok=True)
    (folds_root / f"fold{i+1}" / "labels").mkdir(parents=True, exist_ok=True)


for i, fold in enumerate(folds, start=1):
    fold_img_dir = folds_root / f"fold{i}" / "images"
    fold_lbl_dir = folds_root / f"fold{i}" / "labels"

    for img_path, label_path in fold:
        shutil.copy2(img_path, fold_img_dir / img_path.name)
        shutil.copy2(label_path, fold_lbl_dir / label_path.name)


print("\nDone.\n")
for i, fold in enumerate(folds, start=1):
    print(f"Fold {i}: {len(fold)} samples")