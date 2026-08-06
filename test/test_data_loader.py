import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.data_loader import load_datasets

train_ds, val_ds, test_ds, class_names = load_datasets()

print("=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(f"Number of Classes : {len(class_names)}")
print(f"Classes : {class_names}")

for images, labels in train_ds.take(1):
    print(f"Image Batch Shape : {images.shape}")
    print(f"Label Batch Shape : {labels.shape}")