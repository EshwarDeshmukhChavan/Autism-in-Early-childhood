# app.py
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import os
import time
from PIL import Image

# ==================================
# Page Config
# ==================================
st.set_page_config(
    page_title="ASD Screening Tool",
    page_icon="🧠",
    layout="wide"
)

MODEL_DIR = "models"

# ==================================
# Load Models
# ==================================
@st.cache_resource
def load_all_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory '{MODEL_DIR}' not found. Please add your models there.")
        return models
    for f in os.listdir(MODEL_DIR):
        if f.endswith((".h5", ".keras")):
            name = os.path.splitext(f)[0]
            try:
                models[name] = tf.keras.models.load_model(os.path.join(MODEL_DIR, f))
            except Exception as e:
                st.error(f"Error loading {f}: {e}")
    return models

with st.spinner("Loading AI models..."):
    models = load_all_models()
    model_names = list(models.keys())

# ==================================
# Feature Extractor
# ==================================
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ...]

# ==================================
# Sidebar
# ==================================
st.sidebar.title("⚙️ Controls")

if not model_names:
    st.sidebar.warning("⚠️ No models found in `models/` folder.")
else:
    selected_model_name = st.sidebar.selectbox("Choose a model", model_names)
    model_type = st.sidebar.radio("Select model type:", ["Audio", "Image", "Text/CSV"])

    st.sidebar.markdown("---")
    st.sidebar.info(
        "This tool uses deep learning to analyze infant cries, facial images, "
        "or questionnaire responses. It is **not** a medical diagnosis."
    )

    # ==================================
    # Main Panel
    # ==================================
    st.title("🧠 Multi-Modal ASD Screening Tool")
    model = models[selected_model_name]

    # --- Audio Mode ---
    if model_type == "Audio":
        st.header("🎵 Upload Cry Audio")
        uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])
        if uploaded_file:
            st.audio(uploaded_file, format="audio/wav")
            if st.button("Analyze Audio", type="primary"):
                path = "temp.wav"
                with open(path, "wb") as f: 
                    f.write(uploaded_file.getbuffer())
                features = extract_mfcc(path)
                pred = model.predict(features)[0][0]
                os.remove(path)

                is_asd = pred > 0.5
                conf = pred if is_asd else 1 - pred
                label = "ASD Likely" if is_asd else "Typical"

                st.metric("Prediction", label, f"{conf:.1%} Confidence")
                st.progress(float(conf))

    # --- Image Mode ---
    elif model_type == "Image":
        st.header("🖼️ Upload Facial Image")
        uploaded_img = st.file_uploader("Upload `.jpg` or `.png` image", type=["jpg", "png"])
        if uploaded_img:
            img = Image.open(uploaded_img).resize((128, 128))
            st.image(img, caption="Uploaded Face", use_column_width=True)
            if st.button("Analyze Image", type="primary"):
                x = np.array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                pred = model.predict(x)[0][0]

                is_asd = pred > 0.5
                conf = pred if is_asd else 1 - pred
                label = "ASD Likely" if is_asd else "Typical"

                st.metric("Prediction", label, f"{conf:.1%} Confidence")
                st.progress(float(conf))

    # --- Text/CSV Mode ---
    elif model_type == "Text/CSV":
        st.header("📑 Upload Questionnaire Data")
        uploaded_csv = st.file_uploader("Upload `.csv` file", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.write("Preview of uploaded data:", df.head())
            if st.button("Analyze Questionnaire", type="primary"):
                try:
                    x = df.values.astype("float32")
                    pred = model.predict(x)[0][0]
                    is_asd = pred > 0.5
                    conf = pred if is_asd else 1 - pred
                    label = "ASD Likely" if is_asd else "Typical"

                    st.metric("Prediction", label, f"{conf:.1%} Confidence")
                    st.progress(float(conf))
                except Exception as e:
                    st.error(f"Error predicting: {e}")
