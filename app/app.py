import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# ==========================================
# File Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "threshold.txt"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "creditcard.csv"
)


# ==========================================
# Load Model
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read())

    return model, scaler, threshold


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


# ==========================================
# Load Resources
# ==========================================

try:

    model, scaler, threshold = load_model()
    data = load_data()

except Exception as error:

    st.error("Unable to load model or dataset.")
    st.exception(error)
    st.stop()


# ==========================================
# Header
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Machine learning system for detecting "
    "potentially fraudulent credit card transactions."
)


# ==========================================
# Model Information
# ==========================================

st.success("Random Forest model loaded successfully! ✅")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Model",
        "Random Forest"
    )

with col2:

    st.metric(
        "Decision Threshold",
        f"{threshold:.2f}"
    )

with col3:

    st.metric(
        "Total Transactions",
        f"{len(data):,}"
    )


# ==========================================
# Transaction Prediction
# ==========================================

st.header("🔍 Check a Transaction")

transaction_number = st.number_input(
    "Enter Transaction Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)


if st.button(
    "🚨 Check Transaction",
    use_container_width=True
):

    transaction = (
        data
        .drop("Class", axis=1)
        .iloc[[transaction_number]]
        .copy()
    )

    transaction[["Time", "Amount"]] = (
        scaler.transform(
            transaction[["Time", "Amount"]]
        )
    )

    fraud_probability = (
        model.predict_proba(
            transaction
        )[0][1]
    )

    if fraud_probability >= threshold:

        prediction = "FRAUD"

    else:

        prediction = "NORMAL"


    actual_class = data.iloc[
        transaction_number
    ]["Class"]

    if actual_class == 1:

        actual_result = "FRAUD"

    else:

        actual_result = "NORMAL"


    st.subheader("📊 Transaction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Transaction",
            transaction_number
        )

    with col2:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Actual Class",
            actual_result
        )


    if prediction == "FRAUD":

        st.error("🚨 FRAUD DETECTED")

    else:

        st.success("✅ NORMAL TRANSACTION")


    st.write(
        f"**Prediction:** {prediction}"
    )

    st.write(
        f"**Decision Threshold:** "
        f"{threshold:.2f}"
    )


    with st.expander("View Transaction Details"):

        original_transaction = (
            data.iloc[[transaction_number]]
        )

        st.dataframe(
            original_transaction,
            use_container_width=True
        )


# ==========================================
# Top Suspicious Transactions
# ==========================================

st.header("🚨 Top Suspicious Transactions")

st.write(
    "Transactions with the highest predicted "
    "fraud probability."
)


# ==========================================
# Prepare Dataset for Prediction
# ==========================================

prediction_data = (
    data
    .drop("Class", axis=1)
    .copy()
)

prediction_data[["Time", "Amount"]] = (
    scaler.transform(
        prediction_data[["Time", "Amount"]]
    )
)


# ==========================================
# Calculate Probabilities
# ==========================================

all_probabilities = model.predict_proba(
    prediction_data
)[:, 1]


# ==========================================
# Create Results Table
# ==========================================

suspicious_transactions = pd.DataFrame({

    "Transaction": range(len(data)),

    "Fraud Probability": all_probabilities,

    "Actual Class": data["Class"].values

})


suspicious_transactions["Prediction"] = (
    suspicious_transactions["Fraud Probability"]
    >= threshold
).map({
    True: "FRAUD",
    False: "NORMAL"
})


# ==========================================
# Sort by Fraud Probability
# ==========================================

top_suspicious = (
    suspicious_transactions
    .sort_values(
        "Fraud Probability",
        ascending=False
    )
    .head(10)
    .copy()
)


top_suspicious["Fraud Probability"] = (
    top_suspicious["Fraud Probability"] * 100
).round(2)


top_suspicious["Actual Class"] = (
    top_suspicious["Actual Class"]
    .map({
        0: "NORMAL",
        1: "FRAUD"
    })
)


# ==========================================
# Display Table
# ==========================================

st.dataframe(
    top_suspicious,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# Dataset Statistics
# ==========================================

st.header("📈 Dataset Overview")

normal_count = (
    data["Class"] == 0
).sum()

fraud_count = (
    data["Class"] == 1
).sum()


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Normal Transactions",
        f"{normal_count:,}"
    )

with col2:

    st.metric(
        "Fraud Transactions",
        f"{fraud_count:,}"
    )

with col3:

    st.metric(
        "Fraud Percentage",
        f"{(fraud_count / len(data)) * 100:.4f}%"
    )


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "Credit Card Fraud Detection • "
    "Machine Learning Project"
)