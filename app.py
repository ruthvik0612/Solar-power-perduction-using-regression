import streamlit as st
import pandas as pd
import joblib

# Load the trained Random Forest Model
model = joblib.load("random_forest_model.pkl")

# Streamlit Page Config
st.set_page_config(page_title="Solar Power Prediction App", layout="wide")

# Header
st.title(" Solar Power Prediction App")
st.markdown("""
Welcome to the **Solar Power Generation Prediction App!**  
This tool uses a **Tuned Random Forest Regressor Model** to predict **Power Output (kWh)**  
based on various environmental conditions.
""")

# ---- Replace the input fields + input_data block with this ----

# Sidebar Inputs (add wind_speed)
st.sidebar.header("🔧 Input Features")

temperature = st.sidebar.number_input("Temperature (°C)", 0, 60, 30)
humidity = st.sidebar.number_input("Humidity (%)", 0, 100, 50)
average_pressure = st.sidebar.number_input("Average Pressure (hPa)", 900, 1100, 1013)
average_wind_speed = st.sidebar.number_input("Average Wind Speed (period)", 0.0, 40.0, 5.0)
wind_speed = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)          # <-- NEW
distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon", 0.0, 12.0, 6.0)
sky_cover = st.sidebar.number_input("Sky Cover (0–4)", 0.0, 4.0, 1.0)
visibility = st.sidebar.number_input("Visibility (km)", 0.0, 50.0, 10.0)
wind_direction = st.sidebar.number_input("Wind Direction (°)", 0, 360, 180)

# Prepare Input DataFrame with the exact feature names used during training
input_data = pd.DataFrame({
    'distance-to-solar-noon': [distance_to_solar_noon],
    'temperature': [temperature],
    'wind-direction': [wind_direction],
    'wind-speed': [wind_speed],
    'sky-cover': [sky_cover],
    'visibility': [visibility],
    'humidity': [humidity],
    'average-wind-speed-(period)': [average_wind_speed],
    'average-pressure-(period)': [average_pressure]
})

# Display Input Summary
st.subheader(" Input Summary")
st.dataframe(input_data)

# Predict Power Output
if st.button(" Predict Power Output"):
    try:
        prediction = model.predict(input_data)
        st.success(f" Predicted Power Output: **{prediction[0]:.2f} kWh**")
    except Exception as e:
        st.error(f" Error: {e}")


# About Section
st.sidebar.markdown("---")
st.sidebar.markdown("###  About this Project")
st.sidebar.info("""
 Model: Tuned Random Forest Regressor  
 Dataset: Solar Power Generation Data  
""")