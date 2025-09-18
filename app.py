import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and preprocessing files
model = tf.keras.models.load_model("medical_diagnosis_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder_disease.pkl", "rb") as f:
    label_encoder_disease = pickle.load(f)

with open("label_encoder_prescription.pkl", "rb") as f:
    label_encoder_prescription = pickle.load(f)

st.title("🩺 Medical Diagnosis App")

# User input
patient_problem = st.text_area("Enter patient's problem:")

if st.button("Predict"):
    if patient_problem.strip():
        # Preprocess input
        seq = tokenizer.texts_to_sequences([patient_problem])
        padded = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")

        # Predict
        predictions = model.predict(padded)

        disease_idx = predictions[0].argmax(axis=1)[0]
        prescription_idx = predictions[1].argmax(axis=1)[0]

        disease = label_encoder_disease.inverse_transform([disease_idx])[0]
        prescription = label_encoder_prescription.inverse_transform([prescription_idx])[0]

        st.success(f"🦠 Predicted Disease: **{disease}**")
        st.info(f"💊 Suggested Prescription: **{prescription}**")
    else:
        st.warning("Please enter a patient problem.")
