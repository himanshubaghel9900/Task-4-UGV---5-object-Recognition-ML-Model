from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "object_model.keras"
CLASS_FILE = Path(__file__).parent / "class_names.txt"

CLASS_NAMES = CLASS_FILE.read_text().splitlines()

st.set_page_config(page_title="5-Object Recognition", page_icon="🔍")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image, width, height):
    image = image.convert("RGB")
    image = image.resize((width, height))
    array = np.array(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


st.title("5-Object Recognition")
st.write("Upload a photo of an apple, glass container, oranges, popcorn or water bottle.")

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. "
             "Make sure it is in the same folder as app.py.")
    st.stop()

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

height, width = model.input_shape[1:3]
st.caption(f"Model loaded. It expects {width}x{height} images.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = ImageOps.exif_transpose(Image.open(uploaded))
    st.image(image, caption="Your image", width="stretch")

    batch = preprocess(image, width, height)
    probabilities = model.predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))

    st.success(f"Prediction: {CLASS_NAMES[best]}")
    st.metric("Confidence", f"{probabilities[best] * 100:.1f}%")

    chart_data = pd.DataFrame({"probability": probabilities}, index=CLASS_NAMES)
    st.bar_chart(chart_data)
