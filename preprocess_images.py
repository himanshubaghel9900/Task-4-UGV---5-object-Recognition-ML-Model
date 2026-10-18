
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

# Project and dataset paths
BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"

# Class names and numerical labels
CLASSES = [
    "apple",
    "glass_container",
    "oranges",
    "popcorn",
    "water_bottle",
]

LABELS = {name: index for index, name in enumerate(CLASSES)}

# All images will be resized to 160 x 160 pixels
IMAGE_SIZE = (160, 160)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLITS = ("train", "val", "test")


def load_and_preprocess(split):
    images = []
    labels = []
    class_counts = {name: 0 for name in CLASSES}
    skipped = 0

    split_dir = DATASET_DIR / split

    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing split folder: {split_dir}. "
            "Run split_dataset.py first."
        )

    for class_name in CLASSES:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_dir}"
            )

        image_paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    # Apply orientation stored in the image metadata
                    img = ImageOps.exif_transpose(img)

                    # Convert grayscale/RGBA images to RGB
                    img = img.convert("RGB")

                    # Resize to the required dimensions
                    img = img.resize(
                        IMAGE_SIZE,
                        Image.Resampling.BILINEAR
                    )

                    # Convert to a NumPy array and normalize to [0, 1]
                    array = np.asarray(
                        img, dtype=np.float32
                    ) / 255.0

                images.append(array)
                labels.append(LABELS[class_name])
                class_counts[class_name] += 1

            except (OSError, ValueError) as error:
                skipped += 1
                print(f"Skipped {image_path}: {error}")

    if not images:
        raise ValueError(f"No valid images found in {split_dir}")

    # Shape: (number of images, 160, 160, 3)
    X = np.stack(images).astype(np.float32)

    # Shape: (number of images,)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, class_counts, skipped


def main():
    print("IMAGE PREPROCESSING RESULTS")

    for split in SPLITS:
        X, y, counts, skipped = load_and_preprocess(split)

        print(f"\n{split.upper()} RESULTS")
        print("Images shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Pixel range:", float(X.min()), "to", float(X.max()))
        print("Skipped images:", skipped)

        for class_name, count in counts.items():
            print(f"{class_name}: {count} images")

        # Save processed arrays for future model training
        output_dir = BASE_DIR / "processed_data"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / f"X_{split}.npy", X)
        np.save(output_dir / f"y_{split}.npy", y)

        print(f"Saved processed {split} arrays.")

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
