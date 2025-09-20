import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os

# ======================
# Load Models
# ======================
MODEL_DIR = "models"

@st.cache_resource
def load_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".h5"):
            model_name = f.replace(".h5", "")
            models[model_name] = tf.keras.models.load_model(os.path.join(MODEL_DIR, f))
    return models

models = load_models()
model_names = list(models.keys())

# ======================
# Feature Extraction
# ======================
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.T[np.newaxis, ...]

# ======================
# Streamlit UI
# ======================
st.title("🍼 Infant Cry Classifier (ASD vs TD)")

st.markdown("""
Upload an infant cry `.wav` file, select a trained model, 
and the app will classify whether the cry is from a **Typically Developing (TD)** child or **Autism Spectrum Disorder (ASD)** child.
""")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
selected_model = st.selectbox("Choose a model", model_names)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features
    X = extract_mfcc("temp.wav")

    # Prediction
    model = models[selected_model]
    prob = model.predict(X)[0][0]
    label = "ASD" if prob > 0.5 else "TD"
    confidence = prob if prob > 0.5 else 1 - prob

    st.subheader("Prediction")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
