import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
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
# Constants & Model Loading
# ==================================
MODEL_DIR = "models"
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model_final.h5")
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, "image_model_final.keras")
CSV_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_final.h5")

@st.cache_resource
def load_models():
    """Load all models once and cache."""
    models = {}
    try:
        models['audio_model'] = tf.keras.models.load_model(AUDIO_MODEL_PATH)
        models['image_model'] = tf.keras.models.load_model(IMAGE_MODEL_PATH)
        models['csv_model'] = tf.keras.models.load_model(CSV_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
    return models

with st.spinner("Loading AI models..."):
    models = load_models()
    if not models:
        st.stop()

# ==================================
# Sidebar UI
# ==================================
st.sidebar.title("⚙️ Screening Mode")
app_mode = st.sidebar.selectbox(
    "Choose Screening Method:",
    ["Toddler Questionnaire", "Infant Cry Analysis", "Facial Image Analysis"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer:** This is a screening tool, not a diagnosis. "
    "Consult a healthcare professional for formal evaluation."
)

# ==================================
# Main App UI
# ==================================
st.title("🧠 Multi-Modal ASD Screening Assistant")

# ==================================
# TODDLER QUESTIONNAIRE
# ==================================
if app_mode == "Toddler Questionnaire":
    st.header("Toddler Behaviour Questionnaire")
    st.markdown("Fill the form below; results will be analyzed using the CSV/MLP model.")

    with st.form(key='csv_form'):
        questions = {
            "A1": "Does your child look at you when you call their name?",
            "A2": "How easy is it to get eye contact with your child?",
            "A3": "Does your child point to indicate wants?",
            "A4": "Does your child point to share interest?",
            "A5": "Does your child pretend?",
            "A6": "Does your child follow where you're looking?",
            "A7": "If a family member is upset, does your child try to comfort?",
            "A8": "Describe your child's first words",
            "A9": "Does your child use simple gestures?",
            "A10": "Does your child stare at nothing with no purpose?"
        }
        answers = {q: st.radio(q_text, ['Yes','No'], index=1, horizontal=True, key=q) for q, q_text in questions.items()}

        age_mons = st.number_input("Age in Months", min_value=12, max_value=60, value=24)
        sex = st.selectbox("Sex", ['m','f'])
        jaundice = st.selectbox("Born with Jaundice?", ['yes','no'])
        family_asd = st.selectbox("Family history of ASD?", ['yes','no'])
        who_completed = st.selectbox("Who is completing the test?", ['family member','Health Care Professional','Self','Others'])
        ethnicity = st.selectbox("Ethnicity", ['White European','Asian','Middle Eastern','Black','South Asian','Others','Hispanic','Latino','Mixed','Pacifica'])

        submit_button = st.form_submit_button(label='Submit and Analyze')

    if submit_button:
        with st.spinner("Analyzing questionnaire data..."):
            data = {
                'Age_Mons': age_mons, 'Sex': sex, 'Ethnicity': ethnicity,
                'Jaundice': jaundice, 'Family_mem_with_ASD': family_asd,
                'Who_completed_test': who_completed
            }
            # Convert Yes/No to 1/0
            data.update({k: 1 if v=='Yes' else 0 for k,v in answers.items()})
            df = pd.DataFrame([data])
            # Simple preprocessing: convert categorical to numeric
            df = pd.get_dummies(df)
            # Align columns to model input (add missing columns as zeros)
            input_cols = models['csv_model'].input_shape[1]
            if df.shape[1] < input_cols:
                for _ in range(input_cols - df.shape[1]):
                    df[f'pad_{_}'] = 0
            prediction = models['csv_model'].predict(df.values)
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
            else:
                st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")

# ==================================
# INFANT CRY ANALYSIS
# ==================================
elif app_mode == "Infant Cry Analysis":
    st.header("Infant Cry Audio Analysis")
    st.markdown("Upload a `.wav` file of the infant's cry.")

    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        if st.button("Analyze Cry Audio"):
            with st.spinner("Processing audio..."):
                audio, sr = librosa.load(uploaded_file, sr=22050)
                max_len = sr * 4  # 4 seconds
                if len(audio) < max_len:
                    audio = np.pad(audio, (0, max_len - len(audio)), 'constant')
                else:
                    audio = audio[:max_len]
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
                mfcc = mfcc.T[np.newaxis, ...]  # reshape for model

                prediction = models['audio_model'].predict(mfcc)
                confidence = float(prediction[0][0])
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")

# ==================================
# FACIAL IMAGE ANALYSIS
# ==================================
elif app_mode == "Facial Image Analysis":
    st.header("Facial Image Analysis")
    st.markdown("Upload a facial photograph (`.jpg`, `.jpeg`, `.png`).")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Analyze Facial Image"):
            with st.spinner("Processing image..."):
                img = image.resize((224,224))
                img_array = np.array(img)/255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = models['image_model'].predict(img_array)
                confidence = float(prediction[0][0])
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")
