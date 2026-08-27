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
# 4. Get Fraud Transactions
# ==========================================

fraud_data = data[
    data["Class"] == 1
].head(10)


# ==========================================
# 5. Prepare Transactions
# ==========================================

transactions = fraud_data.drop(
    "Class",
    axis=1
).copy()


transactions[["Time", "Amount"]] = (
    scaler.transform(
        transactions[["Time", "Amount"]]
    )
)


# ==========================================
# 6. Calculate Probabilities
# ==========================================

probabilities = model.predict_proba(
    transactions
)[:, 1]


# ==========================================
# 7. Display Results
# ==========================================

print("Multiple Fraud Transaction Test")
print("=" * 60)

detected = 0

for index, probability in zip(
    fraud_data.index,
    probabilities
):

    if probability >= threshold:
        prediction = "FRAUD"
        detected += 1
    else:
        prediction = "NORMAL"

    print(
        f"Transaction {index} | "
        f"Probability: {probability:.4f} | "
        f"Prediction: {prediction}"
    )


# ==========================================
# 8. Summary
# ==========================================

print("\n" + "=" * 60)

print(f"Fraud Transactions Tested: 10")
print(f"Fraud Transactions Detected: {detected}")
print(f"Fraud Transactions Missed: {10 - detected}")

detection_rate = (detected / 10) * 100

print(
    f"Detection Rate: {detection_rate:.2f}%"
)