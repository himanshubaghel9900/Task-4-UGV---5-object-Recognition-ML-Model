
from pathlib import Path
from PIL import Image
import numpy as np

BASE = Path(__file__).parent
DATASET = BASE / "dataset"

CLASSES = [
    "apple",
    "glass_container",
    "oranges",
    "popcorn",
    "water_bottle",
]

IMAGE_SIZE = (160, 160)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_and_preprocess(split):
    images = []
    labels = []
    skipped = 0

    for label, class_name in enumerate(CLASSES):
        folder = DATASET / split / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            try:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    image = image.resize(
                        IMAGE_SIZE,
                        Image.Resampling.BILINEAR
                    )
                    array = np.asarray(image, dtype=np.float32)
                    array = array / 255.0

                images.append(array)
                labels.append(label)

            except Exception as error:
                skipped += 1
                print(f"Skipped {path.name}: {error}")

    if not images:
        raise ValueError(f"No images loaded from {split}")

    X = np.stack(images)
    y = np.asarray(labels, dtype=np.int64)

    print(f"\n{split.upper()} RESULTS")
    print("Images shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Pixel range:", X.min(), "to", X.max())
    print("Skipped images:", skipped)

    for label, class_name in enumerate(CLASSES):
        print(
            f"{class_name}:",
            int(np.sum(y == label)),
            "images"
        )

    return X, y


if __name__ == "__main__":
    X_train, y_train = load_and_preprocess("train")
    X_val, y_val = load_and_preprocess("val")
    X_test, y_test = load_and_preprocess("test")

    print("\nPreprocessing completed successfully!")
