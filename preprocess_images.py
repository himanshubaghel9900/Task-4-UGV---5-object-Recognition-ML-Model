
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path(__file__).parent
DATASET = BASE / "dataset"

CLASSES = ["apple", "bottle", "capsicum", "cup", "football"]
SPLITS = ["train", "val", "test"]
IMG_SIZE = (160, 160)

def load_and_preprocess(image_path):
    
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)

        
        image_array = np.array(image, dtype=np.float32)

       
        image_array = image_array / 255.0

    return image_array

def load_split(split_name):
    
    images = []
    labels = []

    for label, class_name in enumerate(CLASSES):
        folder = DATASET / split_name / class_name

        if not folder.exists():
            raise FileNotFoundError(
                f"Missing folder: {folder}"
            )

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue

            try:
                array = load_and_preprocess(path)
                images.append(array)
                labels.append(label)
            except Exception as error:
                print(f"Skipping {path.name}: {error}")

    if not images:
        raise ValueError(
            f"No readable images found in {split_name}"
        )

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)

    return X, y

if __name__ == "__main__":
    print("Class labels:")
    for index, name in enumerate(CLASSES):
        print(f"{index}: {name}")

    for split in SPLITS:
        X, y = load_split(split)

        print(f"\n{split.upper()} SET")
        print("Images shape:", X.shape)
        print("Labels shape:", y.shape)
        print("Pixel range:", X.min(), "to", X.max())
        print("Images per class:")

        for index, name in enumerate(CLASSES):
            print(f"  {name}: {np.sum(y == index)}")
