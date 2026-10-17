
from pathlib import Path
import random
import shutil
from PIL import Image

BASE = Path(__file__).parent
SOURCE = BASE / "raw_images"
OUTPUT = BASE / "dataset"

CLASSES = [
    "apple",
    "glass_container",
    "oranges",
    "paper_bag",
    "water_bottle",
]

random.seed(42)

if OUTPUT.exists():
    raise SystemExit(
        "The 'dataset' folder already exists. "
        "Rename it or remove it after backing it up, "
        "then run this script again."
    )

class_images = {}

for class_name in CLASSES:
    folder = SOURCE / class_name

    if not folder.exists():
        raise SystemExit(f"Missing folder: {folder}")

    images = []

    for file in folder.iterdir():
        if not file.is_file():
            continue

        try:
            with Image.open(file) as img:
                img.verify()
            images.append(file)
        except Exception:
            print(f"Skipping invalid image: {file.name}")

    if len(images) != 100:
        raise SystemExit(
            f"{class_name} has {len(images)} valid images, "
            "not 100. Please check this folder."
        )

    random.shuffle(images)
    class_images[class_name] = images

for class_name, images in class_images.items():
    splits = {
        "train": images[:70],
        "val": images[70:85],
        "test": images[85:100],
    }

    for split_name, split_images in splits.items():
        destination = OUTPUT / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

        print(f"{split_name}/{class_name}: {len(split_images)} images")

print("\nDataset split completed successfully!")
print("Training: 350 images")
print("Validation: 75 images")
print("Testing: 75 images")
print("Total: 500 images")
