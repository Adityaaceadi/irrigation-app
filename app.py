
import streamlit as st
import pandas as pd
import joblib
import sklearn

st.title("Irrigation Recommendation System")

st.write("Using scikit-learn version:", sklearn.__version__)

def safe_load_model(path):
    with open(path, "rb") as f:
        return joblib.load(f, mmap_mode=None)

model = safe_load_model("irrigation_model.joblib")

st.write("Enter sensor values:")

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
    pred = model.predict(input_df)[0]
    st.success(f"Recommended irrigation action: {pred}")
