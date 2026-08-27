import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# ==========================================
# Load Model, Scaler and Threshold
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/fraud_detection_model.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    with open(
        "models/threshold.txt",
        "r"
    ) as file:

        threshold = float(file.read())

    return model, scaler, threshold


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "dataset/creditcard.csv"
    )


# ==========================================
# Load Everything
# ==========================================

model, scaler, threshold = load_model()
data = load_data()


# ==========================================
# Title
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Use the trained Machine Learning model "
    "to detect potentially fraudulent credit "
    "card transactions."
)

st.divider()


# ==========================================
# Model Information
# ==========================================

st.subheader("🤖 Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with col2:
    st.metric(
        "Decision Threshold",
        f"{threshold:.1f}"
    )


st.divider()


# ==========================================
# Transaction Selection
# ==========================================

st.subheader("🔎 Select Transaction")

transaction_number = st.number_input(
    "Enter Transaction Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)


# ==========================================
# Prediction Button
# ==========================================

if st.button(
    "🔍 Predict Transaction",
    use_container_width=True
):

    # Convert transaction number to integer

    transaction_number = int(
        transaction_number
    )


    # ======================================
    # Select Transaction
    # ======================================

    transaction = (
        data
        .drop("Class", axis=1)
        .iloc[[transaction_number]]
        .copy()
    )


    # ======================================
    # Scale Time and Amount
    # ======================================

    transaction[
        ["Time", "Amount"]
    ] = scaler.transform(
        transaction[
            ["Time", "Amount"]
        ]
    )


    # ======================================
    # Calculate Fraud Probability
    # ======================================

    fraud_probability = model.predict_proba(
        transaction
    )[0][1]


    # ======================================
    # Make Prediction
    # ======================================

    if fraud_probability >= threshold:

        prediction = "FRAUD"

    else:

        prediction = "NORMAL"


    # ======================================
    # Actual Class
    # ======================================

    actual_class = data.iloc[
        transaction_number
    ]["Class"]


    if actual_class == 1:

        actual_result = "FRAUD"

    else:

        actual_result = "NORMAL"


    # ======================================
    # Display Results
    # ======================================

    st.divider()

    st.subheader("📊 Prediction Result")


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )

    with col2:

        st.metric(
            "Threshold",
            f"{threshold:.2f}"
        )


    st.write(
        f"**Transaction Number:** "
        f"{transaction_number}"
    )

    st.write(
        f"**Model Prediction:** "
        f"{prediction}"
    )

    st.write(
        f"**Actual Class:** "
        f"{actual_result}"
    )


    # ======================================
    # Prediction Message
    # ======================================

    if prediction == "FRAUD":

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

    else:

        st.success(
            "✅ TRANSACTION APPEARS NORMAL"
        )


    # ======================================
    # Correct / Incorrect
    # ======================================

    if prediction == actual_result:

        st.success(
            "✅ Model prediction matches "
            "the actual class."
        )

    else:

        st.warning(
            "⚠️ Model prediction does not "
            "match the actual class."
        )