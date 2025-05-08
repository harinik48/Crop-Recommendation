import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('crop_pred_model.pkl')  # Replace this path with the location of your model

# Set up the Streamlit app layout
st.title('Crop Prediction App')

# Create input fields for the user
temperature = st.number_input("Enter Temperature (°C)", min_value=-50, max_value=50)
humidity = st.number_input("Enter Humidity (%)", min_value=0, max_value=100)
ph = st.number_input("Enter pH value of soil", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Enter Rainfall (mm)", min_value=0, max_value=500)

# When the "Predict" button is pressed
if st.button('Predict'):
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })
    
    # Make predictions using the trained model
    prediction = model.predict(input_data)

    # Show the prediction to the user
    st.write(f"The predicted crop is: {prediction[0]}")  # Adjust this based on the output format of your model
