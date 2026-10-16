
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Five object classes
classes = ["apple", "bottle", "cup", "football", "capsicum"]

# Dataset splits
splits = ["train", "val", "test"]

for split in splits:
    for class_name in classes:
        folder = BASE_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

print("Dataset folders created successfully!")
print(f"Dataset location: {BASE_DIR}")

for split in splits:
    print(f"\n{split.upper()}")
    for class_name in classes:
        print(f"  {class_name}/")
