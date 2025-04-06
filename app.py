import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import gdown
import os
from PIL import Image

# Force TensorFlow to use CPU to avoid Streamlit Cloud GPU errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define file paths
MODEL_PATH = "stacked_model.keras"
LABELS_PATH = "class_labels.json"
CSV_PATH = "medicinal_uses(2).csv"
MODEL_URL = "https://drive.google.com/uc?id=1Ollcw9FIVoKABTraEhPEuxbyg2hCmtjS"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load model without optimizer issues
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Load class labels with error handling
try:
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_labels = json.load(f)
except Exception as e:
    st.error(f"Error loading labels.json: {e}")
    class_labels = {}

# Load medicinal uses CSV with error handling
try:
    df = pd.read_csv(CSV_PATH, encoding="latin1")
except Exception as e:
    st.error(f"Error loading medicinal_uses.csv: {e}")
    df = pd.DataFrame()

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Plant Classification & Remedies")
st.write("Upload an image to classify and get medicinal remedies.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    with st.spinner("Processing image..."):
        # Preprocess and predict
        image = Image.open(uploaded_file)
        #processed_image = preprocess_image(image)
        #prediction = model.predict(processed_image)
        #predicted_index = np.argmax(prediction)
        class_name = class_labels[predicted_index]
    
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write(f"### Predicted Class: {class_name}")
    
    # Display medicinal uses if available
    if not df.empty:
        utilities = df.iloc[predicted_index]["Utilities"]
        remedies = df.iloc[predicted_index]["Remedies"]
        st.write("### Medicinal Uses:")
        #st.write(utilities)
        st.write("### Remedies:")
        #st.write(remedies)
    else:
        st.write("### No medicinal data available for this class.")

elif uploaded_file is not None:
    st.error("Model could not be loaded. Please try again.....")
# Button to clear cache and restart app
if st.button("Clear Cache & Restart"):
    st.experimental_rerun()

# Free memory after prediction
del model
tf.keras.backend.clear_session()
