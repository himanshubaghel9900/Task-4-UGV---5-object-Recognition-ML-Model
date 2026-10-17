import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
PROJECT_DIR = Path.home() / "ML_Project"
REPO_DIR = PROJECT_DIR / "github_data"

MODEL_PATH = REPO_DIR / "object_model.keras"
TEST_DIR = REPO_DIR / "dataset" / "test"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "apple",
    "glass_container",
    "oranges",
    "popcorn",
    "water_bottle"
]

print("Model path:", MODEL_PATH)
print("Test directory:", TEST_DIR)
print("Classes:", CLASS_NAMES)

test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=(128, 128),
    batch_size=32,
    shuffle=False
)

print("Class order:", test_data.class_names)
y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_data
])

y_prob = model.predict(test_data, verbose=1)

y_pred = np.argmax(y_prob, axis=1)

print("Actual labels:", y_true.shape)
print("Predictions:", y_pred.shape)
print("Prediction Scores:", y_prob.shape)
accuracy = accuracy_score(y_true, y_pred)

correct = int(np.sum(y_true == y_pred))
incorrect = int(np.sum(y_true != y_pred))

print(f"Total test images: {len(y_true)}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Test accuracy: {accuracy:.2%}")
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

report_path = RESULTS_DIR / "classification_report.txt"
report_path.write_text(report)

print("Report saved to:", report_path)
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(9, 7))

disp.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)

plt.title("Confusion Matrix - UGV Object Classification")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
plt.show()
# Calculate metrics for each class
precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES)),
    zero_division=0
)

# Set positions for the bars
x = np.arange(len(CLASS_NAMES))
width = 0.25

# Create the graph
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - width, precision, width, label="Precision")
ax.bar(x, recall, width, label="Recall")
ax.bar(x + width, f1, width, label="F1-score")

# Format the graph
ax.set_title("UGV Object Classification - Model Performance")
ax.set_xlabel("Object Class")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha="right")
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

# Save the graph
plt.savefig(RESULTS_DIR / "class_performance.png", dpi=300)
plt.show()

print("Graph saved to:", RESULTS_DIR / "class_performance.png")
accuracy = np.mean(y_true == y_pred) * 100

fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(["Test Accuracy"], [accuracy])

ax.set_title("UGV Object Classification - Test Accuracy")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)

ax.text(
    0,
    accuracy + 2,
    f"{accuracy:.2f}%",
    ha="center",
    fontsize=12
)

ax.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()

plt.savefig(RESULTS_DIR / "test_accuracy.png", dpi=300)
plt.show()

print(f"Test accuracy: {accuracy:.2f}%")
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

img = load_img(image_path, target_size=(128, 128))
img_array = img_to_array(img)

plt.imshow(img)
plt.title("Image Selected for Prediction")
plt.axis("off")
plt.show()

print("Image shape:", img_array.shape)
import numpy as np

# Add a batch dimension to the image
input_batch = np.expand_dims(img_array, axis=0)

# Predict the class
probabilities = model.predict(input_batch, verbose=0)[0]
predicted_index = int(np.argmax(probabilities))

print("Predicted class:", CLASS_NAMES[predicted_index])
print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
print("Actual class: apple")
