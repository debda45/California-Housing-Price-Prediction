import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# 1. Page Configuration
st.set_page_config(page_title="California Housing Predictor", page_icon="🏡", layout="centered")

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('xgboost_housing_model.pkl')

model = load_model()

# 3. Dashboard Header
st.title("🏡 California Housing Price Predictor")
st.markdown("""
This dashboard uses an **XGBoost Regression** model to predict median house values in California districts. 
Adjust the features in the sidebar to see how they impact the estimated property value!
""")

# 4. Sidebar for User Inputs
st.sidebar.header("Input Property Features")

def user_input_features():
    MedInc = st.sidebar.slider("Median Income (in tens of thousands)", 0.5, 15.0, 3.8)
    HouseAge = st.sidebar.slider("House Age (Years)", 1.0, 52.0, 28.0)
    AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.4)
    AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    Population = st.sidebar.number_input("Population in Block", min_value=10, max_value=10000, value=1425)
    AveOccup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.sidebar.slider("Latitude", 32.5, 42.0, 35.6)
    Longitude = st.sidebar.slider("Longitude", -124.3, -114.3, -119.5)
    
    data = {
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'AveRooms': AveRooms,
        'AveBedrms': AveBedrms,
        'Population': Population,
        'AveOccup': AveOccup,
        'Latitude': Latitude,
        'Longitude': Longitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# 5. Display User Inputs
st.subheader("Selected Features")
st.write(input_df)

# 6. Prediction Logic
if st.button("Predict Price"):
    # The model predicts the log of the price
    log_prediction = model.predict(input_df)[0]
    
    # Inverse the log transformation: exp(y) - 1
    actual_prediction = np.expm1(log_prediction)
    
    # The dataset target is in units of $100,000
    final_price_usd = actual_prediction * 100000
    
    st.success(f"### Predicted Median House Value: ${final_price_usd:,.2f}")
    
    st.info("Note: The model heavily weighs Location (Latitude/Longitude) and Median Income to make this prediction.")