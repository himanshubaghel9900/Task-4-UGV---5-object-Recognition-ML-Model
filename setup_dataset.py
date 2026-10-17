
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET = BASE_DIR / "dataset"

classes = ["apple", "glass_container", "popcorn", "", "water_bottle"]
splits = ["train", "val", "test"]

for split in splits:
    for class_name in classes:
        folder = DATASET / split / class_name
        folder.mkdir(parents=True, exist_ok=True)

print("Dataset folders created successfully!")
print(f"Dataset location: {DATASET}")

for split in splits:
    print(f"\n{split.upper()}")
    for class_name in classes:
        print(f"  {class_name}/")
