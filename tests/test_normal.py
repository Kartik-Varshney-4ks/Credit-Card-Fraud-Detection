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
# 4. Get Normal Transactions
# ==========================================

normal_data = data[
    data["Class"] == 0
].head(10)


# ==========================================
# 5. Prepare Transactions
# ==========================================

transactions = normal_data.drop(
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

print("Multiple Normal Transaction Test")
print("=" * 60)

correct = 0

for index, probability in zip(
    normal_data.index,
    probabilities
):

    if probability >= threshold:
        prediction = "FRAUD"
    else:
        prediction = "NORMAL"

    if prediction == "NORMAL":
        correct += 1

    print(
        f"Transaction {index} | "
        f"Probability: {probability:.4f} | "
        f"Prediction: {prediction}"
    )


# ==========================================
# 8. Summary
# ==========================================

print("\n" + "=" * 60)

print("Normal Transactions Tested: 10")
print(f"Normal Transactions Correctly Identified: {correct}")
print(f"Normal Transactions Incorrectly Flagged: {10 - correct}")

accuracy = (correct / 10) * 100

print(
    f"Normal Transaction Accuracy: {accuracy:.2f}%"
)