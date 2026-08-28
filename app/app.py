import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
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


model, scaler, threshold = load_model()
data = load_data()


# ==========================================
# Header
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Machine Learning application for detecting "
    "potentially fraudulent credit card transactions."
)

st.divider()


# ==========================================
# Dashboard Statistics
# ==========================================

st.subheader("📊 Dataset Overview")

total_transactions = len(data)

normal_transactions = (
    data["Class"] == 0
).sum()

fraud_transactions = (
    data["Class"] == 1
).sum()


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )


with col2:

    st.metric(
        "Normal Transactions",
        f"{normal_transactions:,}"
    )


with col3:

    st.metric(
        "Fraud Transactions",
        f"{fraud_transactions:,}"
    )


with col4:

    st.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )


st.divider()


# ==========================================
# Model Information
# ==========================================

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)


with col1:

    st.info(
        "**Algorithm**\n\n"
        "Logistic Regression"
    )


with col2:

    st.info(
        "**Fraud Detection**\n\n"
        "Binary Classification"
    )


with col3:

    st.info(
        "**Features**\n\n"
        "30 Input Features"
    )


st.divider()


# ==========================================
# Transaction Selection
# ==========================================

st.subheader("🔎 Transaction Prediction")

st.write(
    "Select a transaction from the dataset "
    "to test the trained model."
)


transaction_number = st.number_input(
    "Transaction Number",
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
    # Save Original Amount
    # ======================================

    original_amount = (
        transaction["Amount"].iloc[0]
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
    # Fraud Probability
    # ======================================

    fraud_probability = model.predict_proba(
        transaction
    )[0][1]


    # ======================================
    # Prediction
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
    # Results
    # ======================================

    st.divider()

    st.subheader("📈 Prediction Result")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2%}"
        )


    with col2:

        st.metric(
            "Transaction Amount",
            f"${original_amount:.2f}"
        )


    with col3:

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
    # Probability Bar
    # ======================================

    st.write("### Fraud Probability")

    st.progress(
        float(fraud_probability)
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
            "the actual transaction class."
        )

    else:

        st.warning(
            "⚠️ Model prediction does not "
            "match the actual transaction class."
        )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Credit Card Fraud Detection | "
    "Python + Scikit-learn + Streamlit"
)