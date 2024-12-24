import streamlit as st
import numpy as np
import pickle

# Load the model and scaler
with open('stroke-prediction-model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title and description
st.title("Stroke Prediction App")
st.write("""
This app predicts the likelihood of a stroke based on health parameters. Please provide the following details:
""")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)
hypertension = st.selectbox("Hypertension", ["No (0)", "Yes (1)"])
heart_disease = st.selectbox("Heart Disease", ["No (0)", "Yes (1)"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, step=0.1, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, step=0.1, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# Map categorical inputs to numerical values
gender_map = {"Male": [1, 0], "Female": [0, 0], "Other": [0, 1]}
ever_married_map = {"No": 0, "Yes": 1}
residence_type_map = {"Urban": 0, "Rural": 1}
work_type_map = {
    "Private": [0, 1, 0, 0],
    "Self-employed": [0, 0, 1, 0],
    "Govt_job": [0, 0, 0, 0],
    "children": [0, 0, 0, 1],
    "Never_worked": [1, 0, 0, 0]
}
smoking_status_map = {
    "never smoked": [0, 1, 0],
    "formerly smoked": [1, 0, 0],
    "smokes": [0, 0, 1]
}

# Prepare the input data
input_data = [
    age,
    int(hypertension.split()[1][1]),
    int(heart_disease.split()[1][1]),
    ever_married_map[ever_married],
    residence_type_map[residence_type],
    avg_glucose_level,
    bmi
] + gender_map[gender] + work_type_map[work_type] + smoking_status_map[smoking_status]

input_data = np.array([input_data])

# Scale the input data
scaled_data = scaler.transform(input_data)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(scaled_data)[0]

    # Display result
    if prediction == 1:
        st.success("The model predicts a high likelihood of stroke.")
    else:
        st.success("The model predicts a low likelihood of stroke.")
