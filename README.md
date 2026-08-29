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
| Pandas | Data loading and analysis |
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
```

The `Time` and `Amount` features are standardized using `StandardScaler`.

---

## 🔬 Exploratory Data Analysis

Exploratory data analysis was performed using Jupyter Notebook.

The analysis includes:

- Dataset shape
- Column information
- Missing-value analysis
- Class distribution
- Fraud percentage
- Transaction amount analysis
- Feature distributions
- Correlation analysis
- Visualization of normal and fraudulent transactions

The notebook is available in:

```text
notebooks/fraud_detection_eda.ipynb
```

---

## 🤖 Machine Learning Models

Two machine learning models were evaluated.

### 1. Logistic Regression

Logistic Regression was trained using balanced class weights:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)
```

### 2. Random Forest

A Random Forest classifier was also trained and evaluated.

The two models were compared using validation F1 score.

---

## ⚖️ Model Comparison

The validation results were:

| Model | Validation F1 Score |
|---|---:|
| Logistic Regression | 0.6667 |
| Random Forest | **0.7970** |

Based on validation F1 score, **Random Forest** was selected as the final model.

---

## 🎚️ Threshold Selection

Instead of using the default classification threshold of `0.50`, the project evaluates different probability thresholds.

The threshold was selected using validation performance.

### Selected Threshold

```text
0.64
```

This threshold provides a balance between precision and recall for the fraud detection problem.

---

## 📈 Final Model Performance

The final Random Forest model was evaluated on the test dataset.

| Metric | Result |
|---|---:|
| Accuracy | **99.95%** |
| Precision | **98.18%** |
| Recall | **72.97%** |
| F1 Score | **83.72%** |
| Average Precision | **81.07%** |
| Decision Threshold | **0.64** |

### Confusion Matrix

```text
[[42647     1]
 [   20    54]]
```

This means:

- `42,647` normal transactions were correctly classified
- `1` normal transaction was incorrectly classified as fraud
- `54` fraudulent transactions were correctly detected
- `20` fraudulent transactions were missed

---

## 💾 Saved Model Files

After training, the project generates the following files locally:

```text
models/
├── fraud_detection_model.pkl
├── scaler.pkl
└── threshold.txt
```

### Files

**`fraud_detection_model.pkl`**

Contains the trained Random Forest model.

**`scaler.pkl`**

Contains the fitted `StandardScaler`.

**`threshold.txt`**

Contains the selected fraud probability threshold.

These generated files are excluded from Git tracking.

---

## 🔎 Transaction Prediction

The prediction script allows a user to enter a transaction number and receive a prediction.

Run:

```powershell
python src\predict.py
```

Example:

```text
Enter transaction number: 541
```

The program displays:

```text
Fraud Probability
Decision Threshold
Prediction
Actual Class
```

Example result:

```text
Prediction         : FRAUD
Actual Class       : FRAUD
```

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application.

Run:

```powershell
python -m streamlit run app\app.py
```

The application allows users to:

- Select a transaction
- Calculate fraud probability
- View the prediction
- Compare prediction with the actual class
- View transaction details
- View highly suspicious transactions
- View dataset statistics

The Streamlit application provides a user-friendly interface for demonstrating the fraud detection model.

---

## 🧪 Testing

Automated tests are included using **pytest**.

Run:

```powershell
python -m pytest
```

The tests verify:

- Model file availability
- Scaler availability
- Threshold availability
- Model loading
- Scaler loading
- Threshold validity
- Dataset availability
- Model prediction
- Fraud probability validity

Test file:

```text
tests/test_model.py
```

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_detection_model.pkl
│   ├── scaler.pkl
│   └── threshold.txt
│
├── notebooks/
│   └── fraud_detection_eda.ipynb
│
├── src/
│   ├── load_data.py
│   ├── predict.py
│   └── train_model.py
│
├── tests/
│   └── test_model.py
│
├── .gitignore
├── README.md
└── venv/
```

> **Note:** `dataset/`, generated model files, and `venv/` are kept locally and excluded from Git tracking.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Kartik-Varshney-4ks/Credit-Card-Fraud-Detection.git
```

### 2. Move into the project directory

```bash
cd Credit-Card-Fraud-Detection
```

### 3. Create a virtual environment

```powershell
python -m venv venv
```

### 4. Activate the virtual environment

For PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit pytest jupyter
```

---

## ▶️ Usage

### Load and inspect the dataset

```powershell
python src\load_data.py
```

### Train the model

```powershell
python src\train_model.py
```

### Test a transaction

```powershell
python src\predict.py
```

### Run the Streamlit application

```powershell
python -m streamlit run app\app.py
```

### Run automated tests

```powershell
python -m pytest
```

---

## 🚨 Dataset Setup

The dataset is excluded from Git because it is large.

Place the dataset locally at:

```text
dataset/creditcard.csv
```

The project expects the original Credit Card Fraud Detection dataset containing the required columns.

---

## 🔮 Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- XGBoost comparison
- LightGBM comparison
- SMOTE-based class balancing
- Precision-Recall curve visualization
- ROC curve visualization
- Feature importance visualization
- Real-time transaction input
- Improved Streamlit UI
- Model monitoring
- API deployment
- Docker deployment
- Cloud deployment

---

## 📌 Project Status

### Completed

- [x] Dataset loading
- [x] Exploratory data analysis
- [x] Train/validation/test split
- [x] Feature scaling
- [x] Logistic Regression
- [x] Random Forest
- [x] Model comparison
- [x] Threshold optimization
- [x] Final model evaluation
- [x] Model saving
- [x] Transaction prediction
- [x] Streamlit application
- [x] Automated tests
- [x] GitHub project setup
- [x] GitHub repository cleanup
- [x] Project documentation

---

## 📊 Final Result

The project successfully developed a **Random Forest-based credit card fraud detection system**.

Final test performance:

```text
Accuracy          : 99.95%
Precision         : 98.18%
Recall            : 72.97%
F1 Score          : 83.72%
Average Precision : 81.07%
Threshold         : 0.64
```

The system can detect potentially fraudulent transactions, make individual transaction predictions, and provide an interactive Streamlit interface.

---

## 👨‍💻 Author

**Kartik Varshney**

Engineering Student

GitHub: [Kartik-Varshney-4ks](https://github.com/Kartik-Varshney-4ks)

This project was independently developed as a machine learning
project for learning, experimentation, and portfolio development.

---

## © Copyright

**Copyright © 2026 Kartik Varshney. All Rights Reserved.**

This project and its source code are the original work of Kartik Varshney.

The repository is publicly available for viewing and educational
reference. Copying, redistributing, modifying, or submitting this
project as another person's original work is not permitted without
prior written permission.

If this project is referenced or used for learning, the original
repository and author must be clearly credited.