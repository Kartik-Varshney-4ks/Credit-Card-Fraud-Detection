import pandas as pd


# ==========================================
# Load Credit Card Fraud Dataset
# ==========================================

def load_data():

    file_path = "dataset/creditcard.csv"

    data = pd.read_csv(file_path)

    return data


# ==========================================
# Main Program
# ==========================================

if __name__ == "__main__":

    data = load_data()

    print("Credit Card Fraud Detection Dataset")
    print("=" * 50)

    print("\nDataset Shape:")
    print(data.shape)

    print("\nFirst 5 Transactions:")
    print(data.head())

    print("\nColumn Names:")
    print(data.columns.tolist())

    print("\nClass Distribution:")
    print(data["Class"].value_counts())

    print("\nClass Distribution Percentage:")
    print(
        data["Class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(4)
    )