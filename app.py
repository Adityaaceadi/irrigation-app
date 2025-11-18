import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title("Irrigation Recommendation System (Cloud Safe Version)")

@st.cache_resource
def load_and_train():
    df = pd.read_csv("irrigation_dataset_100.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, feature_names = load_and_train()

st.subheader("Enter sensor values:")

data = {}
for col in feature_names:
    data[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([data])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    st.success(f"Recommended irrigation action: {pred}")
