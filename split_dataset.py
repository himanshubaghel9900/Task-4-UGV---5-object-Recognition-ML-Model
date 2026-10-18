
from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw_images"
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = [
    "apple",
    "glass_container",
    "oranges",
    "popcorn",
    "water_bottle",
]

# Dataset split proportions
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducible random split
random.seed(42)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # Check the proportions
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

    # Check all source folders before changing anything
    for class_name in CLASSES:
        class_dir = RAW_DIR / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        images = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if len(images) != 1000:
            raise ValueError(
                f"{class_name} has {len(images)} images, expected 1000."
            )

    # Refuse to overwrite an existing dataset
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        raise FileExistsError(
            f"{DATASET_DIR} already contains files. "
            "Rename or back it up before creating a new split."
        )

    # Create split folders
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            (DATASET_DIR / split / class_name).mkdir(
                parents=True, exist_ok=True
            )

    total_counts = {"train": 0, "val": 0, "test": 0}

    for class_name in CLASSES:
        images = [
            p for p in (RAW_DIR / class_name).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        images.sort()
        random.shuffle(images)

        train_end = int(len(images) * TRAIN_RATIO)
        val_end = train_end + int(len(images) * VAL_RATIO)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        print(f"\n{class_name}:")

        for split, files in splits.items():
            destination = DATASET_DIR / split / class_name

            for image_path in files:
                # Copy originals; do not delete raw images
                shutil.copy2(image_path, destination / image_path.name)

            total_counts[split] += len(files)
            print(f"  {split}: {len(files)} images")

    print("\nDataset split completed!")
    print(f"Training images:   {total_counts['train']}")
    print(f"Validation images: {total_counts['val']}")
    print(f"Testing images:    {total_counts['test']}")
    print(f"Total images:      {sum(total_counts.values())}")
    print(f"Output folder: {DATASET_DIR}")


if __name__ == "__main__":
    main()
