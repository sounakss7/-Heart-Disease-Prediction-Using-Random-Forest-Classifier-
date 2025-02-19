import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
st.title("Heart Disease Prediction using Random Forest")

def load_data():
    dataset = pd.read_csv("heart.csv")
    return dataset

df = load_data()

# Data Overview
st.subheader("Dataset Overview")
st.write(df.head())
st.write("Shape of dataset:", df.shape)

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
if st.checkbox("Show Dataset Summary"):
    st.write(df.describe())

# Visualization
st.subheader("Feature Distribution")
feature = st.selectbox("Select a feature to visualize", df.columns[:-1])
fig, ax = plt.subplots()
sns.histplot(df[feature], kde=True, ax=ax)
st.pyplot(fig)

# Train Model
st.subheader("Train Random Forest Model")
test_size = st.slider("Test size (as a fraction)", 0.1, 0.5, 0.2)
n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)

x = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

# Prediction Section
st.subheader("Make a Prediction")
user_input = {}
for col in x.columns:
    user_input[col] = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.write("Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")