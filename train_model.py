import tensorflow as tf
from pathlib import Path


tf.random.set_seed(42) 
BASE = Path(__file__).parent
DATASET = BASE / "dataset"

CLASSES = ["apple", "glass_container", "oranges", "popcorn", "water_bottle"]

trainData = tf.keras.utils.image_dataset_from_directory( DATASET/"train",class_names = CLASSES, image_size=(128, 128), batch_size=30)

valData = tf.keras.utils.image_dataset_from_directory(DATASET / "val", class_names =CLASSES, image_size=(128, 128), batch_size=30, shuffle=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(trainData, validation_data=valData, epochs=20)

model.save("object_model.keras")

with open(BASE / "class_names.txt", "w") as file:
    file.write("\n".join(CLASSES))

print("Training completed!")
