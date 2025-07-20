#Streamlit App for UI -

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

st.title("🌸 Iris Flower Predictor")
st.write("Enter values below to predict the Iris species.")

# Sidebar Inputs
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)

# Output
st.subheader("Input values")
st.write(input_df)

st.subheader("Prediction")
st.write(f"🌼 Predicted Species: **{target_names[prediction]}**")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.bar_chart(proba_df.T)

# Feature Importance
st.subheader("Model Feature Importance")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values(by='importance', ascending=False)

fig, ax = plt.subplots()
sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
st.pyplot(fig)
