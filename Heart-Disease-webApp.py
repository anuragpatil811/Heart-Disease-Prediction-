
import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('C:/Users/Admin/Downloads/train_model.sav', 'rb'))

# Prediction function
def heart_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "✅ The person does **not** have a heart disease."
    else: 
        return "⚠️ The person **has** a heart disease."

# Main function for Streamlit UI
def main():
    st.title('❤️ Heart Disease Prediction Web App')

    # Input fields
    Age = st.text_input('Age of the person')
    Sex = st.text_input('Gender (1 = Male, 0 = Female)')
    CP = st.text_input('Chest pain type (0-3)')
    trestbps = st.text_input('Resting blood pressure (mm Hg)') 
    chol = st.text_input('Serum cholestoral (mg/dl)')
    fbs = st.text_input('Fasting blood sugar > 120 mg/dl (1 = True; 0 = False)')
    restecg = st.text_input('Resting electrocardiographic results (0-2)') 
    thalach = st.text_input('Maximum heart rate achieved')
    exang = st.text_input('Exercise induced angina (1 = Yes; 0 = No)')
    oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    ca = st.text_input('Number of major vessels (0-3) colored by fluoroscopy')
    thal = st.text_input('Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversable Defect)')

    diagnosis = ''

    if st.button('🧠 Predict Heart Disease'):
        try:
            # Convert all inputs to float
            input_data = [
                float(Age), float(Sex), float(CP), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            diagnosis = heart_prediction(input_data)

        except ValueError:
            diagnosis = "❌ Please enter valid **numerical** values in all fields."

    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()
