import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ==========================================
# 1. Load Dataset
# ==========================================

data = pd.read_csv("dataset/creditcard.csv")

print("Dataset Shape:")
print(data.shape)


# ==========================================
# 2. Separate Features and Target
# ==========================================

X = data.drop("Class", axis=1)
y = data["Class"]


# ==========================================
# 3. Create Training + Temporary Data
# ==========================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Create Validation + Test Data
# ==========================================

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining Data:")
print(X_train.shape)

print("\nValidation Data:")
print(X_val.shape)

print("\nTesting Data:")
print(X_test.shape)


# ==========================================
# 5. Scale Time and Amount
# ==========================================

scaler = StandardScaler()

X_train[["Time", "Amount"]] = scaler.fit_transform(
    X_train[["Time", "Amount"]]
)

X_val[["Time", "Amount"]] = scaler.transform(
    X_val[["Time", "Amount"]]
)

X_test[["Time", "Amount"]] = scaler.transform(
    X_test[["Time", "Amount"]]
)


# ==========================================
# 6. Create Balanced Logistic Regression
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 7. Train Model
# ==========================================

model.fit(X_train, y_train)

print("\nModel Training Complete!")


# ==========================================
# 8. Get Validation Probabilities
# ==========================================

val_probability = model.predict_proba(X_val)[:, 1]


# ==========================================
# 9. Find Best Threshold
# ==========================================

thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

best_threshold = 0
best_f1 = 0

print("\nValidation Results:")
print("-" * 60)

for threshold in thresholds:

    y_val_pred = (val_probability >= threshold).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.1f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print("\nBest Threshold:")
print(best_threshold)

print("\nBest Validation F1 Score:")
print(f"{best_f1:.4f}")


# ==========================================
# 10. Final Test Prediction
# ==========================================

test_probability = model.predict_proba(X_test)[:, 1]

y_test_pred = (
    test_probability >= best_threshold
).astype(int)


# ==========================================
# 11. Final Evaluation
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_test_pred
)

precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0
)


print("\nFinal Test Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# 12. Confusion Matrix
# ==========================================

print("\nFinal Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))


# ==========================================
# 13. Save Model
# ==========================================

joblib.dump(
    model,
    "models/fraud_detection_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

with open("models/threshold.txt", "w") as file:
    file.write(str(best_threshold))


print("\nModel, scaler, and threshold saved successfully!")