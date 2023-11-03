import numpy as np
import pickle
import streamlit as st

# Loading the trained model
model = None
with open('./model.pkl', 'rb') as f:
	model = pickle.load(f)

def predict(data):
	input_data = np.asarray(data).reshape(1, -1)
	print("Input_Data: ", input_data)

	prediction = model.predict(input_data)[0]
	print(prediction)

	if(prediction == 0):
		return "The person is not diabetic"
	else:
		return "The person is diabetic"

def main():
	st.title('Diabetes Prediction App')

	pregnancies = st.text_input('Number of Pregnancies')
	glucose = st.text_input('Glucose Level')
	blood_pressure = st.text_input('Blood Pressure Level')
	skin_thickness = st.text_input('Skin Thickness')
	insulin = st.text_input('Insulin Dosages')
	bmi = st.text_input('BMI Level')
	diabetes_pedigree = st.text_input('Diabetes Pedigree')
	age = st.text_input('Age')

	# Make prediction on button click
	diagnosis = ""

	if st.button('Diabetes Test Result'):
		diagnosis = predict([
			float(pregnancies),
			float(glucose),
			float(blood_pressure),
			float(skin_thickness),
			float(insulin),
			float(bmi),
			float(diabetes_pedigree),
			float(age)
		])

	st.success(diagnosis)

if __name__ == "__main__":
	main()