import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Solar Power Generation Predictor", layout="centered")

st.title("☀️ Solar Power Generation Predictor")
st.write("Predict solar power generation in kilowatts (kW)")

model = joblib.load("model.pkl")
df = pd.read_csv("solarpowergeneration.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("🔢 Enter Input Values")

input_data = {}

for col in df.drop("power-generated", axis=1).columns:
    input_data[col] = st.number_input(
        f"{col}",
        value=float(df[col].mean()),
        min_value=0.0
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict Power Generation"):
    prediction = model.predict(input_df)[0]
    prediction = abs(prediction)

    st.success(f"⚡ Predicted Power Generated: {prediction:.2f} kW")
