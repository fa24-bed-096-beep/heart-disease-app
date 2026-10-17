import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# App Title & Layout
st.set_page_config(page_title="Heart Disease Prediction System", layout="centered")
st.title("Heart Disease Prediction System 🫀")
st.write("Enter the patient details below to predict the likelihood of Heart Disease.")

# Load Trained Model (.pkl file)
@st.cache_resource
def load_model():
    return pickle.load(open('heart_svc_model.pkl', 'rb'))

model = load_model()

# Setup Encoders
gender_encoder = LabelEncoder()
gender_encoder.fit(['Male', 'Female'])

# User Inputs (4 Selected Attributes)
st.subheader("Patient Attributes")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1.0, max_value=120.0, value=50.0, step=1.0)
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100.0, max_value=600.0, value=200.0, step=1.0)
    max_hr = st.number_input("Maximum Heart Rate (thalach)", min_value=60.0, max_value=220.0, value=150.0, step=1.0)

# Prediction Button
if st.button("Predict Heart Disease", type="primary"):
    encoded_gender = gender_encoder.transform([gender])[0]
    
    # Input Feature Vector
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [encoded_gender],
        'Cholesterol': [chol],
        'MaxHR': [max_hr]
    })
    
    prediction = model.predict(input_df)[0]
    
    st.divider()
    if prediction == 1:
        st.error("### Result: HEART DISEASE DETECTED")
        st.write("The model predicts a high probability of heart disease presence based on the provided inputs.")
    else:
        st.success("### Result: NO HEART DISEASE DETECTED")
        st.write("The model predicts no significant presence of heart disease based on the provided inputs.")