
import streamlit as st
import pandas as pd
import joblib

st.title("Irrigation Recommendation System")

model = joblib.load("irrigation_model.joblib")

st.write("Enter sensor values to get irrigation recommendation:")

def user_input():
    data = {}
    data["humidity"] = st.number_input("humidity", value=0.0)
    data["soil_moisture"] = st.number_input("soil_moisture", value=0.0)
    data["temperature"] = st.number_input("temperature", value=0.0)
    data["rainfall_last_24h"] = st.number_input("rainfall_last_24h", value=0.0)
    data["day_of_week"] = st.number_input("day_of_week", value=0.0)

    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Recommended action: {prediction}")
