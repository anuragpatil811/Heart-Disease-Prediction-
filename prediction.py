import numpy as np
import pickle 

# Load the saved model correctly
with open('C:/Users/Admin/Downloads/train_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Input data
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_arrays = np.asarray(input_data)

# Reshape because sklearn expects 2D array for single sample
input_data_reshaped = input_data_as_numpy_arrays.reshape(1, -1)

# Use the loaded model to predict
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print("The person does not have a heart disease")
else:
    print("The person has a heart disease")
