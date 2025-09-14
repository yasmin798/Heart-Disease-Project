import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

model_path = "D:/Heart_Disease_Project/models/final_model.pkl"
try:
    pipeline = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

st.title("Heart Disease Prediction")
st.write("Enter your health data below to predict the likelihood of heart disease (0-4 scale).")

st.sidebar.header("User Input Features")

features = {
    "thalach": (-3.0, 2.0),  
    "slope_2": [0, 1],  
    "thal_7.0": [0, 1], 
    "cp_4": [0, 1], 
    "oldpeak": (-1.0, 3.0),  
    "exang_1": [0, 1]  
}

user_data = {}
for feature, values in features.items():
    if isinstance(values, list) and len(values) == 2 and values == [0, 1]:  # Binary
        user_data[feature] = st.sidebar.selectbox(f"{feature}", [0, 1])
    else:  # Numerical
        min_val, max_val = values
        user_data[feature] = st.sidebar.slider(f"{feature} (scaled)", min_val, max_val, (min_val + max_val) / 2)

input_df = pd.DataFrame([user_data])

if st.sidebar.button("Predict"):
    try:
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)
        st.subheader("Prediction Result")
        st.write(f"Predicted Heart Disease Class: **{int(prediction[0])}** (0 = No Disease, 1-4 = Increasing Severity)")
        st.write("Probability Distribution:")
        for i, prob in enumerate(probability[0]):
            st.write(f"Class {i}: {prob:.4f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.subheader("Heart Disease Trends")
train_data_path = "D:/Heart_Disease_Project/data/heart_disease_train.csv"
train_df = pd.read_csv(train_data_path)
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=train_df, x="target", ax=ax)
ax.set_title("Distribution of Heart Disease Classes in Training Data")
ax.set_xlabel("Heart Disease Class")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Thalach vs. Oldpeak Trend")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=train_df, x="thalach", y="oldpeak", hue="target", palette="viridis", ax=ax2)
ax2.set_title("Thalach vs. Oldpeak by Heart Disease Class")
ax2.set_xlabel("Thalach")
ax2.set_ylabel("Oldpeak")
st.pyplot(fig2)
