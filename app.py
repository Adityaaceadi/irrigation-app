
import streamlit as st
import joblib
import numpy as np

st.title("Irrigation Recommendation System")

model = joblib.load("model.joblib")
le = joblib.load("label_encoder.joblib")

humidity = st.number_input("Humidity (%)", 0, 100, 50)
soil = st.number_input("Soil Moisture (%)", 0, 100, 30)
temp = st.number_input("Temperature (°C)", -10, 60, 25)
rain = st.number_input("Rainfall Last 24h (mm)", 0, 500, 5)
day = st.number_input("Day of Week (0-6)", 0, 6, 3)

if st.button("Predict"):
    inp = np.array([[humidity, soil, temp, rain, day]])
    pred = model.predict(inp)[0]
    label = le.inverse_transform([pred])[0]
    st.success(f"Recommended Action: **{label}**")
