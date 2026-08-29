import pandas as pd
import joblib


# ==========================================
# 1. File Paths
# ==========================================

MODEL_PATH = "models/fraud_detection_model.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.txt"
DATA_PATH = "dataset/creditcard.csv"


# ==========================================
# 2. Load Model
# ==========================================

model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read())


# ==========================================
# 3. Display Header
# ==========================================

print("==========================================")
print("       CREDIT CARD FRAUD DETECTION")
print("==========================================")

print("\nModel loaded successfully!")
print(f"Decision Threshold: {threshold}")


# ==========================================
# 4. Load Dataset
# ==========================================

data = pd.read_csv(DATA_PATH)


print(
    f"\nAvailable transactions: "
    f"0 to {len(data) - 1}"
)


# ==========================================
# 5. Get Transaction Number
# ==========================================

try:

    transaction_number = int(
        input("Enter transaction number: ")
    )

except ValueError:

    print("\nInvalid input!")
    print("Please enter a whole number.")

    exit()


# ==========================================
# 6. Validate Transaction Number
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
# 7. Select Transaction
# ==========================================

transaction = (
    data
    .drop("Class", axis=1)
    .iloc[[transaction_number]]
    .copy()
)


# ==========================================
# 8. Scale Time and Amount
# ==========================================

transaction[["Time", "Amount"]] = (
    scaler.transform(
        transaction[["Time", "Amount"]]
    )
)


# ==========================================
# 9. Calculate Fraud Probability
# ==========================================

fraud_probability = (
    model.predict_proba(
        transaction
    )[0][1]
)


# ==========================================
# 10. Make Prediction
# ==========================================

if fraud_probability >= threshold:

    prediction = "FRAUD"

else:

    prediction = "NORMAL"


# ==========================================
# 11. Display Prediction
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
    f"Decision Threshold : "
    f"{threshold:.2f}"
)

print(
    f"Prediction         : "
    f"{prediction}"
)


# ==========================================
# 12. Display Actual Class
# ==========================================

actual_class = data.iloc[
    transaction_number
]["Class"]


if actual_class == 1:

    actual_result = "FRAUD"

else:

    actual_result = "NORMAL"


print(
    f"Actual Class       : "
    f"{actual_result}"
)

print("==========================================")