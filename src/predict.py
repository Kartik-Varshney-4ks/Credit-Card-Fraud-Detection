import pandas as pd
import joblib


# ==========================================
# 1. Load Saved Model
# ==========================================

model = joblib.load("models/fraud_detection_model.pkl")

scaler = joblib.load("models/scaler.pkl")


# ==========================================
# 2. Load Selected Threshold
# ==========================================

with open("models/threshold.txt", "r") as file:
    threshold = float(file.read())


print("Fraud Detection Model Loaded!")
print(f"Decision Threshold: {threshold}")


# ==========================================
# 3. Load Dataset
# ==========================================

data = pd.read_csv("dataset/creditcard.csv")


# ==========================================
# 4. Select One Transaction
# ==========================================

transaction = data.drop("Class", axis=1).iloc[[0]].copy()


# ==========================================
# 5. Scale Time and Amount
# ==========================================

transaction[["Time", "Amount"]] = scaler.transform(
    transaction[["Time", "Amount"]]
)


# ==========================================
# 6. Predict Fraud Probability
# ==========================================

probability = model.predict_proba(transaction)[0][1]


# ==========================================
# 7. Make Final Prediction
# ==========================================

if probability >= threshold:
    prediction = "FRAUD"
else:
    prediction = "NORMAL"


# ==========================================
# 8. Display Result
# ==========================================

print("\nTransaction Prediction:")
print(f"Fraud Probability: {probability:.4f}")
print(f"Prediction: {prediction}")