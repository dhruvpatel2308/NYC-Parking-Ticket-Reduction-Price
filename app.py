import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

st.title('Predicting Petrol Consumptions')
st.markdown('Predict Your Petrol Consumptiosn Based on Petrol Tax (in Cents), Average Income (Dollars), Paved Highways (in Miles), and the Proportion of the Population with a Driver’s License.')

data = pd.read_csv("data/petrol_consumption.csv")

st.header("Predict")
col1, col2 = st.columns(2)

with col1:
    st.text("Petrol Tax and Income")
    petroltax = st.slider('Petrol Tax (6 to 10)', 1, 10, 6)
    income = st.slider('Income (3000 to 5500)', 1000, 5500, 3000)

with col2:
    st.text("Paved Highways and Driver's Licence Proportion")
    highways = st.slider('Paved Highways (1000 to 18000)', 1000, 18000, 1000)
    dl = st.slider('Proportion of the Population With a Driver’s License (0.5 to 0.8)', 0.30, 0.5, 0.8)
st.text('')
if st.button("Predict Petrol Consumptions"):
    result = predict(np.array([[petroltax, income, highways, dl]]))  # Pass only the 'Level' for prediction
    st.text(f'Predicted Salary: {result[0]} Gallons')

st.text('')
st.text('')
st.markdown('Created by Jerry')
