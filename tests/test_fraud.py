import pandas as pd
import joblib


# ==========================================
# 1. Load Saved Model
# ==========================================

model = joblib.load(
    "models/fraud_detection_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# ==========================================
# 2. Load Threshold
# ==========================================

with open("models/threshold.txt", "r") as file:
    threshold = float(file.read())


# ==========================================
# 3. Load Dataset
# ==========================================

data = pd.read_csv(
    "dataset/creditcard.csv"
)


# ==========================================
# 4. Find First Fraud Transaction
# ==========================================

fraud_transactions = data[
    data["Class"] == 1
]

transaction_number = fraud_transactions.index[0]

print("Testing Fraud Transaction")
print(f"Transaction Number: {transaction_number}")


# ==========================================
# 5. Select Transaction
# ==========================================

transaction = (
    data
    .drop("Class", axis=1)
    .iloc[[transaction_number]]
    .copy()
)


# ==========================================
# 6. Scale Time and Amount
# ==========================================

transaction[["Time", "Amount"]] = (
    scaler.transform(
        transaction[["Time", "Amount"]]
    )
)


# ==========================================
# 7. Calculate Fraud Probability
# ==========================================

fraud_probability = (
    model.predict_proba(transaction)[0][1]
)


# ==========================================
# 8. Make Prediction
# ==========================================

if fraud_probability >= threshold:
    prediction = "FRAUD"
else:
    prediction = "NORMAL"


# ==========================================
# 9. Display Result
# ==========================================

print("\nPrediction Result:")
print(f"Fraud Probability: {fraud_probability:.4f}")
print(f"Decision Threshold: {threshold}")
print(f"Model Prediction: {prediction}")
print("Actual Class: FRAUD")