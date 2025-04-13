import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import gdown
import os
from PIL import Image

# Force TensorFlow to use CPU to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# File paths and URLs
MODEL_PATH = "stacked_model.keras"
LABELS_PATH = "class_labels.json"
CSV_PATH = "medicinal_uses(2).csv"
MODEL_URL = "https://drive.google.com/uc?id=1Ollcw9FIVoKABTraEhPEuxbyg2hCmtjS"

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    # Load model without compiling (saves memory)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_labels():
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return {}

@st.cache_data
def load_csv():
    try:
        return pd.read_csv(CSV_PATH, encoding="latin1")
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# --- UI Starts ---
st.title("🌿 Plant Classification & Remedies")
st.write("Upload a plant image to classify and see its medicinal uses.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        model = load_model()
        class_labels = load_labels()
        df = load_csv()

        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            predicted_index = np.argmax(prediction)
            class_name = class_labels[predicted_index]

        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.subheader(f"🧪 Predicted Class: {class_name}")

        if not df.empty and predicted_index < len(df):
            utilities = df.iloc[predicted_index]["Utilities"]
            remedies = df.iloc[predicted_index]["Remedies"]
            st.write("### 🌱 Medicinal Uses:")
            st.write(utilities)
            st.write("### 🩺 Remedies:")
            st.write(remedies)
        else:
            st.warning("No data available for this class.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

# Button to clear cache and restart app
if st.button("🔄 Clear Cache & Restart"):
    st.cache_resource.clear()
    st.cache_data.clear()
    st.experimental_rerun()

