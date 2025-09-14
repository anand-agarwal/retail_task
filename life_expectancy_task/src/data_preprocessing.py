import pandas as pd
import numpy as np

def preprocess_data(data_path):
    """
    Reads and preprocesses the Life Expectancy dataset.
    Retains spaces in column names (as in the raw CSV),
    handles missing values, encodes categorical variables,
    and builds grouped indices with normalization.
    """
    # -----------------------------
    # Load dataset (retain spaces)
    # -----------------------------
    df = pd.read_csv(data_path)

    # -----------------------------
    # Handle missing values
    # -----------------------------
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # -----------------------------
    # Encode categorical "Status"
    # -----------------------------
    if "Status" in df.columns:
        df["Status_binary"] = df["Status"].map({"Developing": 0, "Developed": 1})
        df = df.drop(columns=["Status"])
    if "Country" in df.columns:
        df = df.drop(columns=["Country"])

    # -----------------------------
    # Define variable groups (retain spaces exactly)
    # -----------------------------
    health_vars = ["under-five deaths ", "Adult Mortality", " BMI ", " thinness  1-19 years"]
    economic_vars = [
        "Income composition of resources", "Schooling", "Status_binary",
        "Alcohol", "percentage expenditure", "Total expenditure",
        "GDP", "Population"
    ]
    disease_vars = ["Hepatitis B", "Measles ", "Polio", "Diphtheria ", " HIV/AIDS"]

    # -----------------------------
    # Normalize inside groups before averaging
    # -----------------------------
    def normalized_mean(df, cols):
        sub = df[cols].copy()
        sub = (sub - sub.mean()) / sub.std(ddof=0)  # z-score per column
        return sub.mean(axis=1)

    df["Health_Index"] = normalized_mean(df, health_vars)
    df["Economic_Index"] = normalized_mean(df, economic_vars)
    df["Disease_Index"] = normalized_mean(df, disease_vars)

    return df


def standardize(X):
    """
    Z-score standardization: (X - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std


def load_data(data_path, features, target, scale=True):
    """
    Loads preprocessed data and selects features + target.
    Optionally applies standardization to features.
    """
    df = preprocess_data(data_path)

    # Select features and target
    X = df[features].values
    y = df[target].values

    if scale:
        X = standardize(X)

    return X, y


# -----------------------------
# Debugging helper
# -----------------------------
if __name__ == "__main__":
    data_path = "/Users/anandagarwal/retail_task/life_expectancy_task/data/Life Expectancy.csv"
    df = preprocess_data(data_path)
    print("Preprocessed DataFrame (first 5 rows):")
    print(df[["Health_Index", "Economic_Index", "Disease_Index", "Life expectancy "]].head())

    X, y = load_data(
        data_path,
        features=["Health_Index", "Economic_Index", "Disease_Index"],
        target="Life expectancy ",
        scale=True
    )
    print("X (standardized) shape:", X.shape)
    print("y shape:", y.shape)
    print("First 5 rows of standardized X:\n", X[:5])
