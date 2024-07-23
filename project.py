import streamlit as st
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# Load the trained model and scaler
model_path = './random_forest_model.pkl'
loaded_model = pickle.load(open(model_path, 'rb'))
scaler_path = './standard_scaler.pkl'  # Assuming you saved the scaler during training
scaler = pickle.load(open(scaler_path, 'rb'))

def predict_wqi_rf(model, scaler, input_values):
    """
    Predicts WQI based on input feature values using a trained Random Forest model.

    Parameters:
    - model: Trained RandomForestRegressor model
    - scaler: StandardScaler used for feature scaling during training
    - input_values: List or array containing input values for each feature

    Returns:
    - Predicted WQI value
    """
    # Ensure input_values is a 2D array
    input_values = np.array(input_values).reshape(1, -1)

    # Scale the input features using the same scaler used during training
    input_scaled = scaler.transform(input_values)

    # Make predictions
    wqi_prediction = model.predict(input_scaled)

    return wqi_prediction[0]

# Streamlit app
def main():
    st.title("Water Quality Index Prediction")

    # Add input components for your features
    ph = st.number_input("pH:", min_value=0.0, max_value=14.0, value=7.0)
    do = st.number_input("Dissolved Oxygen (mg/L):", min_value=0.0, max_value=20.0, value=5.0)
    co = st.number_input("Total coliform (MPN/100ml):", min_value=0.0, max_value=20.0, value=5.0)
    bod = st.number_input("Biological Oxygen Demand (mg/l):", min_value=0.0, max_value=20.0, value=5.0)
    ec = st.number_input("Electrical conductivity (µmhos/cm):", min_value=0.0, max_value=20.0, value=5.0)
    na = st.number_input("Nitrate and Nitrite (mg/L):", min_value=0.0, max_value=20.0, value=5.0)
    

    # Example usage:
    new_data = [ph, do, co, bod, ec, na]  
    if st.button("Predict WQI"):
        predicted_wqi_rf = predict_wqi_rf(loaded_model, scaler, new_data)

        # Display the predicted WQI
        st.write(f'Predicted WQI: {predicted_wqi_rf}')

if __name__ == "_main_":
    main()