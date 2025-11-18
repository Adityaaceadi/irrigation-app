import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_and_train():
    df = pd.read_csv("irrigation_dataset_100.csv")
    df.columns = df.columns.str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if df.empty:
        st.error("❌ ERROR: Dataset became empty after cleaning. Check your CSV.")
        st.stop()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns.tolist()

st.title("Irrigation Predictor")

model, feature_names = load_and_train()

inputs = []
for f in feature_names:
    val = st.number_input(f, value=0.0)
    inputs.append(val)

if st.button("Predict"):
    pred = model.predict([inputs])[0]
    st.success(f"Prediction: {pred}")
