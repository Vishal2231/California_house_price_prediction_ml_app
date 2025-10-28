# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Housing Price Predictor", layout="centered")

model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

st.title("🏠 Housing Price Prediction")

# Input fields (use the same feature names used during training)
longitude = st.number_input("Longitude", value=None, placeholder="Enter longitude")
latitude = st.number_input("Latitude", value=None, placeholder="Enter latitude")
housing_median_age = st.number_input("House Age", value=None, placeholder="Enter house age")
total_rooms = st.number_input("Total Rooms", value=None, placeholder="Enter total rooms")
total_bedrooms = st.number_input("Total Bedrooms", value=None, placeholder="Enter total bedrooms")
population = st.number_input("Population", value=None, placeholder="Enter population")
households = st.number_input("Households", value=None, placeholder="Enter households")
median_income = st.number_input("Median Income", value=None, placeholder="Enter median income")
ocean_proximity = st.selectbox("Location", ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"])

if st.button("Predict Price"):
    input_data = pd.DataFrame({
        "longitude": [longitude],
        "latitude": [latitude],
        "housing_median_age": [housing_median_age],
        "total_rooms": [total_rooms],
        "total_bedrooms": [total_bedrooms],
        "population": [population],
        "households": [households],
        "median_income": [median_income],
        "ocean_proximity": [ocean_proximity]
    })
    # transform and predict
    try:
        X = pipeline.transform(input_data)
        pred = model.predict(X)[0]
        st.success(f"Predicted House Price: $ {pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
