import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Diabetes Prediction App")

# Collect user input
preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
plas = st.number_input("Plasma Glucose Concentration", min_value=0.0, value=120.0)
pres = st.number_input("Diastolic Blood Pressure", min_value=0.0, value=70.0)
skin = st.number_input("Triceps Skinfold Thickness", min_value=0.0, value=25.0)
insu = st.number_input("Serum Insulin", min_value=0.0, value=80.0)
mass = st.number_input("BMI", min_value=0.0, value=28.0)
pedi = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

inputs = np.array([[preg, plas, pres, skin, insu, mass, pedi, age]])
inputs_scaled = scaler.transform(inputs)

if st.button("Predict"):
    pred = model.predict(inputs_scaled)[0]
    if pred == 1:
        st.error("The model predicts: Diabetic")
    else:
        st.success("The model predicts: Not Diabetic")
