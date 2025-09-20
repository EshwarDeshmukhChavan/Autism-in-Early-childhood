import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import joblib
from PIL import Image
import os

# ==================================
# Page Configuration
# ==================================
st.set_page_config(
    page_title="Multi-Modal ASD Screening Tool",
    page_icon="🧠",
    layout="wide"
)

# ==================================
# Constants & Asset Loading
# ==================================
MODEL_DIR = "models"
# --- Define all paths for your specific models and preprocessors ---
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "asd_cnn_rnn_final.h5")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "image_model_final.keras")
CSV_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_final.h5")
AUDIO_SCALER_PATH = os.path.join(MODEL_DIR, "audio_scaler.pkl")
CSV_SCALER_PATH = os.path.join(MODEL_DIR, "csv_scaler.pkl")
CSV_COLUMNS_PATH = os.path.join(MODEL_DIR, "csv_columns.joblib")

@st.cache_resource
def load_assets():
    """Loads all models and preprocessors once and caches them."""
    assets = {}
    try:
        # Load models
        assets['audio_model'] = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        assets['image_model'] = tf.keras.models.load_model(IMAGE_MODEL_PATH)
        assets['csv_model'] = tf.keras.models.load_model(CSV_MODEL_PATH)
        
        # Load preprocessors for audio and CSV models
        assets['audio_scaler'] = joblib.load(AUDIO_SCALER_PATH)
        assets['csv_scaler'] = joblib.load(CSV_SCALER_PATH)
        assets['csv_columns'] = joblib.load(CSV_COLUMNS_PATH)
    except Exception as e:
        st.error(f"Error loading assets: {e}. Please ensure all model and preprocessor files exist in the '{MODEL_DIR}' directory.")
        return None
    return assets

# --- Load all assets on startup ---
with st.spinner("Warming up the AI models, please wait..."):
    assets = load_assets()

# ==================================
# Sidebar UI
# ==================================
st.sidebar.title("⚙️ Controls")
app_mode = st.sidebar.selectbox(
    "Choose a Screening Method:",
    ["Toddler Questionnaire", "Infant Cry Analysis", "Facial Image Analysis"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer:** This is a screening tool, not a diagnostic one. "
    "Consult a healthcare professional for a formal diagnosis."
)

# ==================================
# Main App UI
# ==================================
st.title("🧠 Multi-Modal ASD Screening Assistant")

# Stop the app if assets failed to load
if not assets:
    st.stop()

# ==============================================================================
#                      UI FOR TODDLER QUESTIONNAIRE (mlp_model_final)
# ==============================================================================
if app_mode == "Toddler Questionnaire":
    st.header("Toddler Behaviour Questionnaire")
    st.markdown("Please fill out the form below. The data will be analyzed by the `mlp_model_final` model.")

    with st.form(key='csv_form'):
        st.subheader("Behavioural Questions (A1-A10)")
        questions = {
            "A1": "Does your child look at you when you call his/her name?", "A2": "How easy is it for you to get eye contact with your child?",
            "A3": "Does your child point to indicate that s/he wants something?", "A4": "Does your child point to share interest with you?",
            "A5": "Does your child pretend?", "A6": "Does your child follow where you’re looking?",
            "A7": "If you or someone else in the family is visibly upset, does your child show signs of wanting to comfort them?",
            "A8": "Would you describe your child’s first words as:", "A9": "Does your child use simple gestures?",
            "A10": "Does your child stare at nothing with no apparent purpose?"
        }
        
        # Invert logic: A "No" to a typical developmental milestone is a potential indicator (1)
        answers = {q_key: 1 - st.radio(q_text, ('Yes', 'No'), index=1, horizontal=True, key=q_key) for q_key, q_text in questions.items()}

        st.subheader("Demographic Information")
        c1, c2 = st.columns(2)
        with c1:
            age_mons = st.number_input("Age in Months", min_value=12, max_value=60, value=24)
            sex = st.selectbox("Sex", ('m', 'f'))
            jaundice = st.selectbox("Born with Jaundice?", ('yes', 'no'))
        with c2:
            family_asd = st.selectbox("Family Member with ASD history?", ('yes', 'no'))
            who_completed = st.selectbox("Who is completing the test?", ('family member', 'Health Care Professional', 'Self', 'Others'))
            ethnicity = st.selectbox("Ethnicity", ('White European', 'asian', 'middle eastern', 'black', 'south asian', 'Others', 'Hispanic', 'Latino', 'mixed', 'Pacifica'))

        submit_button = st.form_submit_button(label='Submit and Analyze')

    if submit_button:
        with st.spinner("Processing data..."):
            raw_data = {
                'Age_Mons': age_mons, 'Sex': sex, 'Ethnicity': ethnicity,
                'Jaundice': jaundice, 'Family_mem_with_ASD': family_asd,
                'Who completed the test': who_completed
            }
            raw_data.update(answers)

            # Preprocessing pipeline
            input_df = pd.DataFrame([raw_data])
            processed_df = pd.get_dummies(input_df)
            final_df = processed_df.reindex(columns=assets['csv_columns'], fill_value=0)
            scaled_data = assets['csv_scaler'].transform(final_df)
            
            # Prediction
            prediction = assets['csv_model'].predict(scaled_data)
            confidence = prediction[0][0]

            st.success("Analysis Complete!")
            if confidence > 0.5:
                st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
            else:
                st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")

# ==============================================================================
#                   UI FOR INFANT CRY ANALYSIS (asd_cnn_rnn_final)
# ==============================================================================
elif app_mode == "Infant Cry Analysis":
    st.header("Infant Cry Audio Analysis")
    st.markdown("Upload a `.wav` file of an infant's cry (up to 4 seconds). The `asd_cnn_rnn_final` model will analyze its acoustic properties.")

    uploaded_file = st.file_uploader("Choose a WAV file...", type="wav")

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Analyze Cry Audio", use_container_width=True, type="primary"):
            with st.spinner("Processing audio..."):
                def preprocess_audio(file):
                    SAMPLE_RATE, MAX_LEN_SECONDS = 22050, 4
                    audio, sr = librosa.load(file, sr=SAMPLE_RATE)
                    max_len_samples = int(MAX_LEN_SECONDS * sr)
                    if len(audio) < max_len_samples:
                        audio = np.pad(audio, (0, max_len_samples - len(audio)), 'constant')
                    else:
                        audio = audio[:max_len_samples]
                    
                    features_list = [
                        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512),
                        librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=2048, hop_length=512),
                        librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512),
                        librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=2048, hop_length=512),
                        librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
                    ]
                    features = np.vstack(features_list).T
                    
                    scaler = assets['audio_scaler']
                    features_scaled = scaler.transform(features)
                    
                    return features_scaled.reshape(1, features_scaled.shape[0], features_scaled.shape[1])

                processed_audio = preprocess_audio(uploaded_file)
                prediction = assets['audio_model'].predict(processed_audio)
                confidence = prediction[0][0]

                st.success("Analysis Complete!")
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")

# ==============================================================================
#                    UI FOR FACIAL IMAGE ANALYSIS (image_model_final)
# ==============================================================================
elif app_mode == "Facial Image Analysis":
    st.header("Facial Image Analysis")
    st.markdown("Upload a facial photograph (`.jpg`, `.jpeg`, `.png`). The `image_model_final` model will analyze it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        
        if st.button("Analyze Facial Image", use_container_width=True, type="primary"):
            with st.spinner("Processing image..."):
                img = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = assets['image_model'].predict(img_array)
                confidence = prediction[0][0]

                st.success("Analysis Complete!")
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")

