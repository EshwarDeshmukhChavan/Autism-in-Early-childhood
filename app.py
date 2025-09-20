import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import time
from PIL import Image

# ==================================
# Page Configuration
# ==================================
st.set_page_config(
    page_title="Infant Cry Screener",
    page_icon="🍼",
    layout="wide"
)

# ==================================
# Constants & Model Loading
# ==================================
MODEL_DIR = "models"

@st.cache_resource
def load_all_models():
    """Loads all .h5 and .keras models from the specified directory."""
    models = {}
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found at '{MODEL_DIR}'. Please create it and add your models.")
        return models
        
    for f in os.listdir(MODEL_DIR):
        if f.endswith((".h5", ".keras")):
            model_name = os.path.splitext(f)[0] # Get name without extension
            try:
                models[model_name] = tf.keras.models.load_model(os.path.join(MODEL_DIR, f))
            except Exception as e:
                st.error(f"Error loading model {f}: {e}")
    return models

# --- Load models once and cache them ---
with st.spinner("Loading AI models..."):
    models = load_all_models()
    model_names = list(models.keys())

# ==================================
# Feature Extraction Functions
# ==================================
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    """Extracts MFCC features from an audio file, matching the training process."""
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # Pad or truncate
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        return mfcc.T[np.newaxis, ...]
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None


def preprocess_image(img, target_size=(128, 128)):
    """Preprocess image file for CNN input."""
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if len(img_array.shape) == 2:  # grayscale
        img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)

# ==================================
# Sidebar UI
# ==================================
st.sidebar.title("⚙️ Controls")
selected_model_name = st.sidebar.selectbox("Choose a model for analysis", model_names)

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App:**\n"
    "This tool uses deep learning to screen for potential ASD traits. "
    "It supports both audio (cry) and image-based models. "
    "This is an **informational aid only, not a medical diagnosis.**"
)

# ==================================
# Main Panel UI
# ==================================
st.title("🍼 Infant Cry & Image ASD Screening Tool")

# --- Detect model type from its name ---
is_audio_model = "audio" in selected_model_name.lower()
is_image_model = "img" in selected_model_name.lower() or "image" in selected_model_name.lower()

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Upload Input")

    uploaded_file = None
    if is_audio_model:
        uploaded_file = st.file_uploader("Upload Cry Audio (.wav)", type=["wav"])
        if uploaded_file: st.audio(uploaded_file, format="audio/wav")
    elif is_image_model:
        uploaded_file = st.file_uploader("Upload Infant Image", type=["jpg", "png", "jpeg"])
        if uploaded_file: st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    else:
        st.warning("Model type not detected. Please name your models with 'audio' or 'img'.")

    analyze_button = st.button("🔍 Analyze", use_container_width=True, type="primary")

with col2:
    st.header("📊 Analysis Result")

    if uploaded_file and analyze_button:
        with st.spinner(f"Analyzing with '{selected_model_name}'..."):
            model = models[selected_model_name]

            if is_audio_model:
                # Save temp file
                temp_file_path = "temp.wav"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                features = extract_mfcc(temp_file_path)
                os.remove(temp_file_path)

                if features is not None:
                    prediction = model.predict(features)[0][0]
                else:
                    prediction = None

            elif is_image_model:
                img = Image.open(uploaded_file).convert("RGB")
                features = preprocess_image(img)
                prediction = model.predict(features)[0][0]

            # --- Show results ---
            if prediction is not None:
                is_asd = prediction > 0.5
                confidence = float(prediction if is_asd else 1 - prediction)

                # Big result card
                if is_asd:
                    st.error("🚨 ASD Traits Likely")
                else:
                    st.success("✅ Typically Developing")

                # Confidence gauge
                st.metric(
                    label="Confidence",
                    value=f"{confidence*100:.2f}%",
                    delta=f"Raw Score: {prediction:.4f}"
                )

                st.progress(confidence)

                with st.expander("More Details"):
                    st.write(f"- **Model Used:** `{selected_model_name}`")
                    st.write(f"- **Prediction Raw Score:** `{prediction:.4f}`")
                    st.write("Scores closer to **1.0** → ASD traits, closer to **0.0** → Normal.")

    else:
        st.info("Upload a file and click **Analyze** to see results.")
