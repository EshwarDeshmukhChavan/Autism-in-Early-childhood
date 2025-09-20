import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import os
import time

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
with st.spinner("Warming up the AI models..."):
    models = load_all_models()
    model_names = list(models.keys())

# ==================================
# Feature Extraction
# ==================================
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    """Extracts MFCC features from an audio file, matching the training process."""
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Normalize the features
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # Pad or truncate to a fixed length
        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]

        # Reshape for model input
        return mfcc.T[np.newaxis, ...]
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# ==================================
# Sidebar UI
# ==================================
st.sidebar.title("⚙️ Controls")
selected_model_name = st.sidebar.selectbox("Choose a model for analysis", model_names)

st.sidebar.markdown("---")
st.sidebar.info(
    "**About this App:**\n"
    "This tool uses deep learning to screen for potential ASD traits based on the acoustic features of an infant's cry. "
    "It is an informational aid and not a medical diagnosis."
)

# ==================================
# Main Panel UI
# ==================================
st.title("🍼 Infant Cry ASD Screening Tool")
st.markdown("Upload an infant cry `.wav` file, and the selected AI model will analyze it.")

# --- Two-column layout ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("📤 Upload Audio")
    uploaded_file = st.file_uploader(
        "Select a .wav file from your device", 
        type=["wav"], 
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        analyze_button = st.button("Analyze Cry", use_container_width=True, type="primary")

with col2:
    st.header("📊 Analysis Result")
    
    # --- Logic for when the button is pressed ---
    if uploaded_file and 'analyze_button' in locals() and analyze_button:
        # Save temp file for librosa to read
        temp_file_path = "temp.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(f"Analyzing with '{selected_model_name}' model..."):
            # Process and predict
            features = extract_mfcc(temp_file_path)
            
            if features is not None:
                model = models[selected_model_name]
                prediction = model.predict(features)[0][0]
                
                # Determine label and confidence
                is_asd = prediction > 0.5
                confidence = prediction if is_asd else 1 - prediction
                result_label = "ASD Traits Likely" if is_asd else "Typically Developing"
                
                # --- Display Results ---
                time.sleep(1) # For a better UX feel
                
                if is_asd:
                    st.metric(label="Prediction Result", value=result_label, delta=f"{confidence:.1%} Confidence", delta_color="inverse")
                else:
                    st.metric(label="Prediction Result", value=result_label, delta=f"{confidence:.1%} Confidence", delta_color="normal")
                
                st.progress(confidence)
                
                with st.expander("More Details"):
                    st.write(f"- **Model Used:** `{selected_model_name}`")
                    st.write(f"- **Raw Prediction Score:** `{prediction:.4f}`")
                    st.write(f"- A score > 0.5 indicates a higher likelihood of ASD traits.")

        # Clean up the temporary file
        os.remove(temp_file_path)

    else:
        # --- Placeholder content before analysis ---
        st.info("Upload an audio file and click 'Analyze Cry' to see the results here.")
        

