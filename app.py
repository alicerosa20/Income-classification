import streamlit as st
import joblib
import pandas as pd

# Load model and unique options
income_classification_model = joblib.load("models/income_classification_model_pipeline.pkl")
unique_options = joblib.load("models/income_classification_unique_options.pkl")

st.set_page_config(page_title="Income Classification", page_icon="💰", layout="centered")

st.title("💰 Income Classification Prediction using ML")

st.image(
    "https://t3.ftcdn.net/jpg/04/46/10/70/360_F_446107040_cgPv9kEr8EGj61zsj5ArXeD3deHwP5az.jpg",
    caption="Human Income Classification",
    width=500,
)

# Input fields
age = st.number_input("Age", min_value=0, max_value=100, value=30)

workclass = st.selectbox("Workclass", unique_options["workclass"])
education = st.selectbox("Education", unique_options["education"])
marital_status = st.selectbox("Marital Status", unique_options["marital-status"])
occupation = st.selectbox("Occupation", unique_options["occupation"])
relationship = st.selectbox("Relationship", unique_options["relationship"])
race = st.selectbox("Race", unique_options["race"])
sex = st.selectbox("Sex", unique_options["sex"])

capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=0, max_value=100, value=40)

native_country = st.selectbox("Native Country", unique_options["native-country"])

# Prediction button
if st.button("Predict Income Classification"):
    # Prepare input as DataFrame with correct column names
    input_df = pd.DataFrame([{
        "age": age,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country
    }])

    prediction = income_classification_model.predict(input_df)[0]

    if prediction == 1:
        st.success("Predicted Income Classification: >50K")
    else:
        st.success("Predicted Income Classification: <=50K")
