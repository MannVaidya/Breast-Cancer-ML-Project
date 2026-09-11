import streamlit as st
import pandas as pd
import joblib


# LOAD MODEL, SCALER AND FEATURE NAMES
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")
feature_names = joblib.load("model/feature_names.pkl")


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)


# TITLE
st.title("🩺 Breast Cancer Prediction")

st.write(
    "Enter the tumor measurement values below "
    "to predict whether the tumor is benign or malignant."
)


# INPUT FEATURES
input_data = {}

for feature in feature_names:

    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )


# PREDICTION
if st.button("Predict"):

    # Create DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0]

    benign_probability = probability[0]
    malignant_probability = probability[1]


    # DISPLAY RESULT
    if prediction == 0:

        st.success("### Prediction: Benign")

    else:

        st.error("### Prediction: Malignant")


    # DISPLAY PROBABILITIES
    st.subheader("Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Benign",
            f"{benign_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Malignant",
            f"{malignant_probability * 100:.2f}%"
        )