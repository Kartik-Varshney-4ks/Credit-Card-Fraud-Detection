# 💳 Credit Card Fraud Detection

A Machine Learning project that detects potentially fraudulent credit card transactions using Python and Scikit-learn.

## 📌 Project Overview

Credit card fraud detection is a binary classification problem where each transaction is classified as either:

- `0` → Normal transaction
- `1` → Fraudulent transaction

This project compares multiple machine learning approaches and selects the best-performing model for fraud detection.

Because fraudulent transactions are extremely rare compared with normal transactions, the project focuses on:

- Handling severe class imbalance
- Feature scaling
- Model comparison
- Probability-based fraud detection
- Classification threshold selection
- Precision and recall evaluation

The final selected model is a **Random Forest Classifier**.

---

## 🎯 Objectives

The main objectives of this project are:

- Load and analyze credit card transaction data
- Understand the imbalance between normal and fraudulent transactions
- Perform exploratory data analysis
- Split the dataset into training, validation, and testing sets
- Scale the `Time` and `Amount` features
- Train a Logistic Regression model
- Train a Random Forest model
- Compare model performance
- Select the best-performing model
- Find an appropriate fraud detection threshold
- Evaluate the final model
- Save the trained model and preprocessing objects
- Build a transaction prediction system
- Build a Streamlit web application
- Add automated tests using pytest

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and data analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Joblib | Saving and loading trained models |
| Streamlit | Web application and dashboard |
| Jupyter Notebook | Data analysis and experimentation |
| VS Code | Development environment |
| Pytest | Automated testing |
| Git & GitHub | Version control |

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset.

The dataset contains:

- `284,807` transactions
- `30` input features
- `1` target column: `Class`

### Class Distribution

| Class | Transactions | Percentage |
|---|---:|---:|
| Normal | 284,315 | 99.8273% |
| Fraud | 492 | 0.1727% |

Fraudulent transactions represent only approximately **0.173%** of the complete dataset.

This severe class imbalance makes accuracy alone unsuitable for evaluating fraud detection performance.

---

## 🔍 Features

The dataset contains:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`

The `Class` column is the target variable.

```text
Class = 0 → Normal
Class = 1 → Fraud