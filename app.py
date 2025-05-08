import streamlit as st
import joblib
import pandas as pd

# Load the trained model and LabelEncoder
model = joblib.load('crop_pred_model.pkl')  # Ensure correct file path
label_encoder = joblib.load('crop_pred_labelencoder.pkl')  # Ensure this is the correct file for label encoding

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

    # Use the label encoder to convert the predicted label back to the crop name
    crop_name = label_encoder.inverse_transform(prediction)

    # Show the prediction to the user
    st.write(f"The predicted crop is: {crop_name[0]}")  # Display the crop name
