# Import necessary libraries
import streamlit as st
import joblib
import numpy as np

# Function to load the pre-trained Random Forest Regressor model
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return joblib.load(model_path)

# Function to make predictions
def predict(model, input_features):
    return model.predict(np.array(input_features).reshape(1, -1))

# Load the pre-trained model
model_path = r"C:\Users\User\Desktop\sbh24\random_forest_model.pkl"
model = load_model(model_path)

# Streamlit app
def main():
    # Title of the app
    st.title('Water Quality Index Predictor')

    # Input fields for user to enter six numerical inputs
    feature1 = st.number_input("pH")
    feature2 = st.number_input("Dissolved Oxygen (mg/L)")
    feature3 = st.number_input("Total coliform (MPN/100ml)")
    feature4 = st.number_input("Biological Oxygen Demand (mg/l)")
    feature5 = st.number_input("Electrical conductivity (µmhos/cm)")
    feature6 = st.number_input("Nitrate and Nitrite (mg/L)")

    # Button to make prediction
    if st.button('Predict'):
        # Collect input features
        input_features = [feature1, feature2, feature3, feature4, feature5, feature6]
        # Make prediction
        prediction = predict(model, input_features)
        # Display prediction
        st.write(f"Predicted Output: {prediction[0]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
