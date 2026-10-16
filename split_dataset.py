
from pathlib import Path
import random
import shutil
from PIL import Image

BASE = Path(__file__).parent
RAW = BASE / "raw_images"
OUTPUT = BASE / "dataset"

CLASSES = ["apple", "bottle", "cup", "football", "capsicum"]
random.seed(42)

for name in CLASSES:
    source = RAW / name

    files = []
    for file in source.iterdir():
        if file.is_file():
            try:
                with Image.open(file) as img:
                    img.verify()
                files.append(file)
            except Exception:
                pass

    random.shuffle(files)

    if len(files) < 20:
        print(f"ERROR: {name} has only {len(files)} valid images.")
        raise SystemExit

    splits = {
        "train": files[:14],
        "val": files[14:17],
        "test": files[17:20],
    }

    for split, selected in splits.items():
        destination = OUTPUT / split / name
        destination.mkdir(parents=True, exist_ok=True)

        for file in selected:
            shutil.copy2(file, destination / file.name)

        print(f"{split}/{name}: {len(selected)} images")

print("\nDataset split completed!")
print("Your dataset is in the 'dataset' folder.")
