import os
import joblib
import pandas as pd


# ==========================================
# File Paths
# ==========================================

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.txt"
DATA_PATH = "dataset/creditcard.csv"


# ==========================================
# Test 1 — Model File Exists
# ==========================================

def test_model_file_exists():

    assert os.path.exists(MODEL_PATH)


# ==========================================
# Test 2 — Scaler File Exists
# ==========================================

def test_scaler_file_exists():

    assert os.path.exists(SCALER_PATH)


# ==========================================
# Test 3 — Threshold File Exists
# ==========================================

def test_threshold_file_exists():

    assert os.path.exists(THRESHOLD_PATH)


# ==========================================
# Test 4 — Model Loads Successfully
# ==========================================

def test_model_loads():

    model = joblib.load(MODEL_PATH)

    assert model is not None


# ==========================================
# Test 5 — Scaler Loads Successfully
# ==========================================

def test_scaler_loads():

    scaler = joblib.load(SCALER_PATH)

    assert scaler is not None


# ==========================================
# Test 6 — Threshold Is Valid
# ==========================================

def test_threshold_is_valid():

    with open(THRESHOLD_PATH, "r") as file:

        threshold = float(file.read())

    assert 0 < threshold < 1


# ==========================================
# Test 7 — Dataset Exists
# ==========================================

def test_dataset_exists():

    assert os.path.exists(DATA_PATH)


# ==========================================
# Test 8 — Model Can Predict
# ==========================================

def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    data = pd.read_csv(DATA_PATH)

    X = data.drop("Class", axis=1).iloc[[0]].copy()

    X[["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    prediction = model.predict(X)[0]

    assert prediction in [0, 1]


# ==========================================
# Test 9 — Fraud Probability Is Valid
# ==========================================

def test_fraud_probability():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    data = pd.read_csv(DATA_PATH)

    X = data.drop("Class", axis=1).iloc[[0]].copy()

    X[["Time", "Amount"]] = scaler.transform(
        X[["Time", "Amount"]]
    )

    probability = model.predict_proba(X)[0][1]

    assert 0 <= probability <= 1