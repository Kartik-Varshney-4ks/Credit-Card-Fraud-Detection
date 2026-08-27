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
# 2. Load Decision Threshold
# ==========================================

with open("models/threshold.txt", "r") as file:
    threshold = float(file.read())


print("==========================================")
print("       CREDIT CARD FRAUD DETECTION")
print("==========================================")

print("\nModel loaded successfully!")
print(f"Decision Threshold: {threshold}")


# ==========================================
# 3. Load Dataset
# ==========================================

data = pd.read_csv(
    "dataset/creditcard.csv"
)


# ==========================================
# 4. Ask for Transaction Number
# ==========================================

print(
    f"\nAvailable transactions: "
    f"0 to {len(data) - 1}"
)

try:

    transaction_number = int(
        input("Enter transaction number: ")
    )

except ValueError:

    print("\nInvalid input!")
    print("Please enter a whole number.")

    exit()


# ==========================================
# 5. Check Transaction Number
# ==========================================

if (
    transaction_number < 0
    or transaction_number >= len(data)
):

    print("\nInvalid transaction number!")

    print(
        f"Please enter a number between "
        f"0 and {len(data) - 1}."
    )

    exit()


# ==========================================
# 6. Select Transaction
# ==========================================

transaction = (
    data
    .drop("Class", axis=1)
    .iloc[[transaction_number]]
    .copy()
)


# ==========================================
# 7. Scale Time and Amount
# ==========================================

transaction[["Time", "Amount"]] = (
    scaler.transform(
        transaction[["Time", "Amount"]]
    )
)


# ==========================================
# 8. Calculate Fraud Probability
# ==========================================

fraud_probability = (
    model.predict_proba(
        transaction
    )[0][1]
)


# ==========================================
# 9. Make Prediction
# ==========================================

if fraud_probability >= threshold:

    prediction = "FRAUD"

else:

    prediction = "NORMAL"


# ==========================================
# 10. Display Prediction
# ==========================================

print("\n==========================================")
print("           TRANSACTION RESULT")
print("==========================================")

print(
    f"Transaction Number : {transaction_number}"
)

print(
    f"Fraud Probability  : "
    f"{fraud_probability:.4f}"
)

print(
    f"Decision Threshold : {threshold}"
)

print(
    f"Prediction         : {prediction}"
)


# ==========================================
# 11. Display Actual Class
# ==========================================

actual_class = data.iloc[
    transaction_number
]["Class"]


if actual_class == 1:

    actual_result = "FRAUD"

else:

    actual_result = "NORMAL"


print(
    f"Actual Class       : {actual_result}"
)

print("==========================================")