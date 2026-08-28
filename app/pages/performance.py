import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
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
# Prepare Features and Target
# ==========================================

X = data.drop(
    "Class",
    axis=1
)

y = data["Class"]


# ==========================================
# Create Train / Validation / Test Split
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


# ==========================================
# Scale Test Data
# ==========================================

X_test = X_test.copy()

X_test[
    ["Time", "Amount"]
] = scaler.transform(
    X_test[
        ["Time", "Amount"]
    ]
)


# ==========================================
# Test Set Predictions
# ==========================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probabilities >= threshold
).astype(int)


# ==========================================
# Calculate Test Metrics
# ==========================================

accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ==========================================
# Page Header
# ==========================================

st.title("📊 Model Performance")

st.write(
    "Evaluation of the trained Logistic Regression "
    "model on the unseen test dataset."
)

st.divider()


# ==========================================
# Dataset Split Information
# ==========================================

st.subheader("📂 Dataset Split")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Training Data",
        f"{len(X_train):,}"
    )


with col2:

    st.metric(
        "Validation Data",
        f"{len(X_validation):,}"
    )


with col3:

    st.metric(
        "Testing Data",
        f"{len(X_test):,}"
    )


st.divider()


# ==========================================
# Evaluation Metrics
# ==========================================

st.subheader("📈 Test Set Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )


with col2:

    st.metric(
        "Precision",
        f"{precision:.2%}"
    )


with col3:

    st.metric(
        "Recall",
        f"{recall:.2%}"
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1:.2%}"
    )


st.divider()


# ==========================================
# Confusion Matrix
# ==========================================

st.subheader("🔢 Confusion Matrix")

cm_dataframe = pd.DataFrame(
    cm,
    index=[
        "Actual Normal",
        "Actual Fraud"
    ],
    columns=[
        "Predicted Normal",
        "Predicted Fraud"
    ]
)


st.dataframe(
    cm_dataframe,
    use_container_width=True
)


# ==========================================
# Confusion Matrix Explanation
# ==========================================

st.write("### What the values mean")

true_negative = cm[0][0]
false_positive = cm[0][1]
false_negative = cm[1][0]
true_positive = cm[1][1]


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.info(
        f"**True Negatives**\n\n"
        f"{true_negative:,}"
    )


with col2:

    st.warning(
        f"**False Positives**\n\n"
        f"{false_positive:,}"
    )


with col3:

    st.warning(
        f"**False Negatives**\n\n"
        f"{false_negative:,}"
    )


with col4:

    st.success(
        f"**True Positives**\n\n"
        f"{true_positive:,}"
    )


st.divider()


# ==========================================
# Prediction Distribution
# ==========================================

st.subheader(
    "📊 Test Set Prediction Distribution"
)


normal_predictions = (
    test_predictions == 0
).sum()

fraud_predictions = (
    test_predictions == 1
).sum()


distribution = pd.DataFrame(
    {
        "Transaction Type": [
            "Normal",
            "Fraud"
        ],
        "Count": [
            normal_predictions,
            fraud_predictions
        ]
    }
)


st.bar_chart(
    distribution.set_index(
        "Transaction Type"
    )
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
        "**Decision Threshold**\n\n"
        f"{threshold:.2f}"
    )


with col3:

    st.info(
        "**Evaluation Dataset**\n\n"
        "Unseen Test Set"
    )


# ==========================================
# Important Note
# ==========================================

st.divider()

st.caption(
    "Metrics shown above are calculated on the "
    "held-out test dataset and are not calculated "
    "using the training data."
)