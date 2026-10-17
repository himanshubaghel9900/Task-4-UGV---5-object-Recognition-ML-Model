from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).parent / "object_model.keras"
CLASS_FILE = Path(__file__).parent / "class_names.txt"
LOW_CONFIDENCE = 0.60

st.set_page_config(page_title="5-Object Recognition", page_icon="🔍")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    lines = CLASS_FILE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip()]


def pretty(name):
    return name.replace("_", " ").title()


def preprocess(image, width, height):
    image = image.convert("RGB")
    image = image.resize((width, height))
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


st.title("5-Object Recognition")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. "
             "Make sure it is in the same folder as app.py.")
    st.stop()

if not CLASS_FILE.exists():
    st.error(f"Class list not found: {CLASS_FILE.name}. "
             "It is created by train_model.py and must be next to app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

class_names = load_class_names()
num_outputs = model.output_shape[-1]
if len(class_names) != num_outputs:
    st.error(f"The model has {num_outputs} outputs but {CLASS_FILE.name} lists "
             f"{len(class_names)} classes. The model and the class list do "
             "not match, so predictions would be wrong. Re-train or use the "
             "files from the same run.")
    st.stop()

height, width = model.input_shape[1:3]
st.write("Upload a photo of one of these: "
         + ", ".join(pretty(name) for name in class_names))
st.caption(f"Model loaded. It expects {width}x{height} images.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded))
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. "
                 "Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Your image", width="stretch")

    batch = preprocess(image, width, height)
    probabilities = model.predict(batch, verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    best = int(order[0])
    second = int(order[1])
    confidence = float(probabilities[best])

    if confidence >= LOW_CONFIDENCE:
        st.success(f"Prediction: {pretty(class_names[best])}")
    else:
        st.warning(
            f"The model is not confident about this image. Its best guess is "
            f"{pretty(class_names[best])} ({confidence * 100:.1f}%), then "
            f"{pretty(class_names[second])} "
            f"({probabilities[second] * 100:.1f}%). "
            "The object may not be one of the classes it was trained on."
        )
    st.metric("Confidence", f"{confidence * 100:.1f}%")

    chart_data = pd.DataFrame(
        {"probability": probabilities},
        index=[pretty(name) for name in class_names],
    )
    st.bar_chart(chart_data)
