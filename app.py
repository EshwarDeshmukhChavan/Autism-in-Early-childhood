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
# Constants & Model Directory
# ==================================
MODEL_DIR = "models"

# ==================================
# Load Models Dynamically
# ==================================
@st.cache_resource
def load_models():
    models = {}
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory '{MODEL_DIR}' not found.")
        return models

    for f in os.listdir(MODEL_DIR):
        if f.endswith((".h5", ".keras")):
            model_name = os.path.splitext(f)[0].lower()  # name in lowercase
            try:
                models[model_name] = tf.keras.models.load_model(os.path.join(MODEL_DIR, f))
            except Exception as e:
                st.error(f"Error loading model {f}: {e}")
    return models

with st.spinner("Loading AI models..."):
    models = load_models()

if not models:
    st.stop()

# ==================================
# Sidebar UI
# ==================================
st.sidebar.title("⚙️ Controls")
app_mode = st.sidebar.selectbox(
    "Choose Screening Method:",
    ["Questionnaire", "Infant Cry Audio", "Facial Image"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ Disclaimer: This is a screening tool, not a diagnostic tool. "
    "Consult a healthcare professional for formal evaluation."
)

# ==================================
# Helper Functions
# ==================================
def extract_mfcc(file_path, n_mfcc=40, max_len=200):
    audio, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T[np.newaxis, ...]

# ==================================
# Main App
# ==================================
st.title("🧠 Multi-Modal ASD Screening Assistant")

# ----------------------------
# 1. Questionnaire Mode
# ----------------------------
if app_mode == "Questionnaire":
    st.header("Toddler Behaviour Questionnaire")
    st.markdown("Fill in the form below for analysis.")

    with st.form(key='questionnaire_form'):
        st.subheader("Behavioural Questions (A1-A10)")
        questions = {
            "A1": "Does your child look at you when you call their name?",
            "A2": "How easy is it to get eye contact with your child?",
            "A3": "Does your child point to indicate that they want something?",
            "A4": "Does your child point to share interest with you?",
            "A5": "Does your child pretend?",
            "A6": "Does your child follow where you are looking?",
            "A7": "If someone is upset, does your child show signs of comfort?",
            "A8": "Would you describe your child's first words as meaningful?",
            "A9": "Does your child use simple gestures?",
            "A10": "Does your child stare at nothing with no purpose?"
        }

        answers = {}
        for key, question in questions.items():
            answer = st.radio(question, ('Yes', 'No'), index=1, horizontal=True, key=key)
            answers[key] = 0 if answer == "Yes" else 1  # Yes=0, No=1

        st.subheader("Demographics")
        age = st.number_input("Age in months", min_value=12, max_value=60, value=24)
        sex = st.selectbox("Sex", ('m', 'f'))
        jaundice = st.selectbox("Born with jaundice?", ('yes', 'no'))
        family_asd = st.selectbox("Family member with ASD history?", ('yes', 'no'))
        ethnicity = st.selectbox("Ethnicity", ['White European', 'Asian', 'Middle Eastern', 'Black',
                                               'South Asian', 'Others', 'Hispanic', 'Latino', 'Mixed', 'Pacifica'])
        who_completed = st.selectbox("Who completed the test?", ['Family Member', 'Health Care Professional', 'Self', 'Others'])

        submit_btn = st.form_submit_button("Submit and Analyze")

    if submit_btn:
        if not csv_columns:
            st.error("CSV columns file not found. Cannot process questionnaire input.")
        else:
            df = pd.DataFrame([{
                **answers,
                "Age_Mons": age,
                "Sex": sex,
                "Ethnicity": ethnicity,
                "Jaundice": jaundice,
                "Family_mem_with_ASD": family_asd,
                "Who completed the test": who_completed
            }])

            # One-hot encode categorical columns
            categorical_cols = ["Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test"]
            df = pd.get_dummies(df, columns=categorical_cols, dtype=float)

            # Reindex to match training columns
            df = df.reindex(columns=csv_columns, fill_value=0)

            # Find CSV/MLP model
            csv_model_key = next((k for k in models.keys() if 'csv' in k or 'mlp' in k), None)
            if csv_model_key:
                model = models[csv_model_key]
                try:
                    pred = model.predict(df.values.astype(np.float32))
                    confidence = float(pred[0][0])
                    if confidence > 0.5:
                        st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                    else:
                        st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")
                except Exception as e:
                    st.error(f"Error predicting: {e}")
            else:
                st.warning("No CSV/Questionnaire model found.")


# ----------------------------
# 2. Infant Cry Audio
# ----------------------------
elif app_mode == "Infant Cry Audio":
    st.header("Infant Cry Audio Analysis")
    st.markdown("Upload a `.wav` file of the infant's cry.")
    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        if st.button("Analyze Cry"):
            temp_path = "temp.wav"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            audio_model_key = next((k for k in models.keys() if 'audio' in k or 'rnn' in k), None)
            if audio_model_key:
                model = models[audio_model_key]
                features = extract_mfcc(temp_path)
                pred = model.predict(features)
                confidence = float(pred[0][0])
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")
            else:
                st.warning("No audio model found.")
            os.remove(temp_path)

# ----------------------------
# 3. Facial Image Analysis
# ----------------------------
elif app_mode == "Facial Image":
    st.header("Facial Image Analysis")
    st.markdown("Upload a facial image (`.jpg`, `.jpeg`, `.png`).")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Analyze Image"):
            image_model_key = next((k for k in models.keys() if 'image' in k or 'cnn' in k), None)
            if image_model_key:
                model = models[image_model_key]
                img = image.resize((224,224))
                arr = tf.keras.preprocessing.image.img_to_array(img)/255.0
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr)
                confidence = float(pred[0][0])
                if confidence > 0.5:
                    st.error(f"Prediction: ASD Traits Likely (Confidence: {confidence:.2%})")
                else:
                    st.success(f"Prediction: ASD Traits Unlikely (Confidence: {1-confidence:.2%})")
            else:
                st.warning("No image model found.")
