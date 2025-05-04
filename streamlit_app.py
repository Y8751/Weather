
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🌦️ هل ستمطر غداً؟ - RainTomorrow Predictor")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv")
    df = df[["MinTemp", "MaxTemp", "Rainfall", "WindGustSpeed", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp3pm", "RainTomorrow"]]
    df.dropna(inplace=True)
    le = LabelEncoder()
    df["RainTomorrow"] = le.fit_transform(df["RainTomorrow"])  # Yes:1, No:0
    return df

df = load_data()

# Train a simple model
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("أدخل البيانات للتنبؤ")

def user_input_features():
    MinTemp = st.sidebar.slider("MinTemp", -10.0, 30.0, 10.0)
    MaxTemp = st.sidebar.slider("MaxTemp", 10.0, 45.0, 25.0)
    Rainfall = st.sidebar.slider("Rainfall", 0.0, 100.0, 5.0)
    WindGustSpeed = st.sidebar.slider("WindGustSpeed", 0.0, 150.0, 30.0)
    Humidity9am = st.sidebar.slider("Humidity9am", 0.0, 100.0, 50.0)
    Humidity3pm = st.sidebar.slider("Humidity3pm", 0.0, 100.0, 50.0)
    Pressure9am = st.sidebar.slider("Pressure9am", 980.0, 1040.0, 1010.0)
    Pressure3pm = st.sidebar.slider("Pressure3pm", 980.0, 1040.0, 1010.0)
    Temp3pm = st.sidebar.slider("Temp3pm", 5.0, 45.0, 22.0)
    data = {
        'MinTemp': MinTemp,
        'MaxTemp': MaxTemp,
        'Rainfall': Rainfall,
        'WindGustSpeed': WindGustSpeed,
        'Humidity9am': Humidity9am,
        'Humidity3pm': Humidity3pm,
        'Pressure9am': Pressure9am,
        'Pressure3pm': Pressure3pm,
        'Temp3pm': Temp3pm
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("🔍 نتيجة التنبؤ:")
st.write("💧 **ستنزل أمطار غداً**" if prediction == 1 else "🌤️ **لن تنزل أمطار غداً**")
st.write(f"نسبة التأكد: {round(prediction_proba[prediction] * 100, 2)}%")
