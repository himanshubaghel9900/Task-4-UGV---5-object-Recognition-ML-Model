from pathlib import Path

import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path(__file__).parent / "object_model.keras"
CLASS_NAMES = ["apple", "bottle", "capsicum", "cup", "football"]

st.set_page_config(page_title="5-Object Recognition", page_icon="🔍")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


st.title("5-Object Recognition")
st.write("Upload a photo of an apple, bottle, capsicum, cup or football.")

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
    image = Image.open(uploaded)
    st.image(image, caption="Your image", use_container_width=True)
