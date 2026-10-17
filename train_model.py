import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt


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

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(trainData, validation_data=valData, epochs=20)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("accuracy_graph.png", dpi=300)
plt.show()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_graph.png", dpi=300)
plt.show()

model.save("object_model.keras")

with open(BASE / "class_names.txt", "w") as file:
    file.write("\n".join(CLASSES))

print("Training completed!")
