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
# Custom Styling
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
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

    model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r"
    ) as file:

        threshold = float(
            file.read()
        )

    return model, scaler, threshold


# ==========================================
# Load Dataset
# ==========================================

@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


# ==========================================
# Calculate All Fraud Probabilities
# ==========================================
# IMPORTANT:
# The "_" before model and scaler tells
# Streamlit not to hash these objects.

@st.cache_data
def calculate_probabilities(
    data,
    _model,
    _scaler
):

    prediction_data = (
        data
        .drop("Class", axis=1)
        .copy()
    )

    prediction_data[
        ["Time", "Amount"]
    ] = _scaler.transform(
        prediction_data[
            ["Time", "Amount"]
        ]
    )

    probabilities = _model.predict_proba(
        prediction_data
    )[:, 1]

    return probabilities


# ==========================================
# Load Resources
# ==========================================

try:

    model, scaler, threshold = load_model()

    data = load_data()

    all_probabilities = calculate_probabilities(
        data,
        model,
        scaler
    )

except Exception as error:

    st.error(
        "Unable to load model or dataset."
    )

    st.exception(error)

    st.stop()


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="main-title">'
    '💳 Credit Card Fraud Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning system for detecting potentially '
    'fraudulent credit card transactions.'
    '</div>',
    unsafe_allow_html=True
)

st.success(
    "🌲 Random Forest model loaded successfully!"
)


# ==========================================
# Dashboard Metrics
# ==========================================

normal_count = int(
    (data["Class"] == 0).sum()
)

fraud_count = int(
    (data["Class"] == 1).sum()
)

fraud_percentage = (
    fraud_count / len(data)
) * 100


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🤖 Model",
        "Random Forest"
    )


with col2:

    st.metric(
        "🎚️ Threshold",
        f"{threshold:.2f}"
    )


with col3:

    st.metric(
        "💳 Transactions",
        f"{len(data):,}"
    )


with col4:

    st.metric(
        "🚨 Fraud Cases",
        f"{fraud_count:,}"
    )


st.divider()


# ==========================================
# Transaction Prediction
# ==========================================

st.header(
    "🔎 Check a Transaction"
)

st.write(
    "Enter a transaction number to calculate "
    "its fraud probability."
)


transaction_number = st.number_input(
    "Transaction Number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)


if st.button(
    "🚨 Check Transaction",
    use_container_width=True
):

    # --------------------------------------
    # Select Transaction
    # --------------------------------------

    transaction = (
        data
        .drop("Class", axis=1)
        .iloc[[transaction_number]]
        .copy()
    )


    # --------------------------------------
    # Scale Transaction
    # --------------------------------------

    transaction[
        ["Time", "Amount"]
    ] = scaler.transform(
        transaction[
            ["Time", "Amount"]
        ]
    )


    # --------------------------------------
    # Get Fraud Probability
    # --------------------------------------

    fraud_probability = float(
        model.predict_proba(
            transaction
        )[0][1]
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    if fraud_probability >= threshold:

        prediction = "FRAUD"

    else:

        prediction = "NORMAL"


    # --------------------------------------
    # Actual Class
    # --------------------------------------

    actual_class = int(
        data.iloc[
            transaction_number
        ]["Class"]
    )


    if actual_class == 1:

        actual_result = "FRAUD"

    else:

        actual_result = "NORMAL"


    # ======================================
    # Transaction Result
    # ======================================

    st.subheader(
        "📊 Transaction Result"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Transaction",
            f"{transaction_number:,}"
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


    # ======================================
    # Fraud Probability Visualization
    # ======================================

    st.subheader(
        "📈 Fraud Probability"
    )

    st.progress(
        min(fraud_probability, 1.0)
    )

    st.caption(
        f"Fraud probability: "
        f"{fraud_probability * 100:.2f}%"
    )


    # ======================================
    # Risk Level
    # ======================================

    if fraud_probability >= threshold:

        st.error(
            "🚨 FRAUD DETECTED"
        )

        st.write(
            "**Prediction:** FRAUD"
        )

        st.write(
            "**Risk Level:** HIGH"
        )


    elif fraud_probability >= 0.30:

        st.warning(
            "⚠️ SUSPICIOUS TRANSACTION"
        )

        st.write(
            "**Prediction:** NORMAL"
        )

        st.write(
            "**Risk Level:** MEDIUM"
        )


    else:

        st.success(
            "✅ NORMAL TRANSACTION"
        )

        st.write(
            "**Prediction:** NORMAL"
        )

        st.write(
            "**Risk Level:** LOW"
        )


    # ======================================
    # Decision Threshold
    # ======================================

    st.info(
        f"Decision threshold: **{threshold:.2f}** "
        f"({threshold * 100:.0f}%)"
    )


    # ======================================
    # Transaction Details
    # ======================================

    with st.expander(
        "🔍 View Transaction Details"
    ):

        original_transaction = (
            data.iloc[
                [transaction_number]
            ]
        )

        st.dataframe(
            original_transaction,
            use_container_width=True,
            hide_index=True
        )


st.divider()


# ==========================================
# Top Suspicious Transactions
# ==========================================

st.header(
    "🚨 Top Suspicious Transactions"
)

st.write(
    "Transactions with the highest predicted "
    "fraud probability."
)


# ==========================================
# Create Results Table
# ==========================================

suspicious_transactions = pd.DataFrame({

    "Transaction": range(
        len(data)
    ),

    "Fraud Probability": all_probabilities,

    "Actual Class": data[
        "Class"
    ].values

})


# ==========================================
# Prediction
# ==========================================

suspicious_transactions[
    "Prediction"
] = (
    suspicious_transactions[
        "Fraud Probability"
    ]
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


# ==========================================
# Format Probability
# ==========================================

top_suspicious[
    "Fraud Probability"
] = (
    top_suspicious[
        "Fraud Probability"
    ] * 100
).round(2)


# ==========================================
# Format Actual Class
# ==========================================

top_suspicious[
    "Actual Class"
] = (
    top_suspicious[
        "Actual Class"
    ]
    .map({

        0: "NORMAL",

        1: "FRAUD"

    })
)


# ==========================================
# Display Suspicious Transactions
# ==========================================

st.dataframe(
    top_suspicious,
    use_container_width=True,
    hide_index=True
)


st.divider()


# ==========================================
# Dataset Overview
# ==========================================

st.header(
    "📊 Dataset Overview"
)


col1, col2, col3, col4 = st.columns(4)


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
        f"{fraud_percentage:.4f}%"
    )


with col4:

    st.metric(
        "Total Features",
        f"{len(data.columns) - 1}"
    )


# ==========================================
# Class Distribution
# ==========================================

st.subheader(
    "📊 Class Distribution"
)


distribution_data = pd.DataFrame({

    "Class": [
        "Normal",
        "Fraud"
    ],

    "Transactions": [
        normal_count,
        fraud_count
    ]

})


st.bar_chart(
    distribution_data.set_index(
        "Class"
    )
)


# ==========================================
# Footer
# ==========================================

st.divider()

st.caption(
    "💳 Credit Card Fraud Detection • "
    "Random Forest Machine Learning Project"
)