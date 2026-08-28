# 💳 Credit Card Fraud Detection

A Machine Learning project that detects potentially fraudulent credit card transactions using Python and Scikit-learn.

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where each transaction is classified as either:

- `0` → Normal transaction
- `1` → Fraudulent transaction

This project uses a Logistic Regression machine learning model to identify potentially fraudulent transactions.

Because fraudulent transactions are extremely rare compared with normal transactions, the project focuses on handling class imbalance and selecting an appropriate prediction threshold.

---

## 🎯 Objectives

The main objectives of this project are:

- Load and analyze credit card transaction data
- Understand the imbalance between normal and fraudulent transactions
- Split the dataset into training, validation, and testing sets
- Scale numerical features
- Train a Logistic Regression model
- Handle class imbalance using balanced class weights
- Select an appropriate classification threshold
- Evaluate the model using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Save the trained model for future predictions
- Test the model on normal and fraudulent transactions
- Build a Streamlit web application
- Display model performance through an interactive dashboard

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Joblib | Saving and loading the trained model |
| Streamlit | Web application and dashboard |
| Jupyter Notebook | Data analysis and experimentation |
| VS Code | Development environment |
| Git & GitHub | Version control |

---

## 📊 Dataset

The project uses the Credit Card Fraud Detection dataset.

The dataset contains:

- `284,807` transactions
- `30` input features
- `1` target column (`Class`)

### Class Distribution

| Class | Transactions |
|---|---:|
| Normal | 284,315 |
| Fraud | 492 |

Fraudulent transactions represent approximately **0.173%** of the dataset.

This severe class imbalance makes fraud detection more challenging.

---

## 🤖 Machine Learning Model

The project uses **Logistic Regression**.

The model is trained using balanced class weights:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)