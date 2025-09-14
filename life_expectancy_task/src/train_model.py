#!/usr/bin/env python3
"""
Gradient Descent Regression (OLS, Ridge, Lasso)
----------------------------------------------
Follows the algorithm:
1. Preprocess Data (assumed done)
2. Choose Regularization (None, Ridge=L2, Lasso=L1)
3. Define Loss (MSE + penalty)
4. Solve Optimization (Gradient Descent)
5. Tune λ (manual/CV, here fixed for demo)
6. Validate on test set
"""

import numpy as np
import pickle
import csv
from data_preprocessing import load_data


# -----------------------------
# Metrics
# -----------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# -----------------------------
# Regression Model (Gradient Descent Only)
# -----------------------------
class RegressionModel:
    def __init__(self, learning_rate=0.01, epochs=1000,
                 regularization=None, lam=0.0, degree=1, feature_names=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization  # None, "ridge", "lasso"
        self.lam = lam
        self.degree = degree
        self.weights = None
        self.feature_names = feature_names

    def _expand_polynomial(self, X):
        """Expand features polynomially up to given degree."""
        if self.degree == 1:
            return X
        X_poly = X
        for d in range(2, self.degree + 1):
            X_poly = np.concatenate([X_poly, X ** d], axis=1)
        return X_poly

    def fit(self, X, y):
        # Add bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        X = self._expand_polynomial(X)
        self.weights = np.zeros(X.shape[1])

        # Gradient descent loop
        for _ in range(self.epochs):
            y_pred = X @ self.weights
            error = y_pred - y
            grad = (X.T @ error) / len(y)

            # Regularization
            if self.regularization == "ridge":
                grad += self.lam * self.weights
                grad[0] -= self.lam * self.weights[0]  # don't regularize bias
            elif self.regularization == "lasso":
                grad += self.lam * np.sign(self.weights)
                grad[0] -= self.lam * np.sign(self.weights[0])

            # Update
            self.weights -= self.learning_rate * grad

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        X = self._expand_polynomial(X)
        return X @ self.weights


# -----------------------------
# Save/Load Utils
# -----------------------------
def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_weights_csv(model, path):
    """Save model weights + corresponding feature names to CSV."""
    if model.feature_names is None:
        feature_names = [f"x{i}" for i in range(len(model.weights) - 1)]
    else:
        feature_names = ["bias"] + model.feature_names

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Weight"])
        for name, w in zip(feature_names, model.weights):
            writer.writerow([name, w])


# -----------------------------
# Main Training Script
# -----------------------------
if __name__ == "__main__":
    # Load processed data
    features = [
        "Status_binary", "Adult Mortality", "Alcohol", "percentage expenditure",
        "Hepatitis B", "Measles ", " BMI ", "under-five deaths ", "Polio",
        "Total expenditure", "Diphtheria ", " HIV/AIDS", "GDP", "Population",
        " thinness  1-19 years", "Income composition of resources", "Schooling", "infant deaths", " thinness 5-9 years"
    ]
    # features = ["Health_Index", "Disease_Index", "Economic_Index"]
    X, y = load_data(
        "data/Life Expectancy.csv",
        features=features,
        target="Life expectancy "
    )

    # Train-test split (80/20)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Model 1: OLS (no regularization)
    model1 = RegressionModel(
        learning_rate=0.001, epochs=5000,
        regularization=None, feature_names=features
    )
    model1.fit(X_train, y_train)
    save_model(model1, "models/model3/ols_model3.pkl")

    # Model 2: Ridge Regression (L2)
    model2 = RegressionModel(
        learning_rate=0.001, epochs=5000,
        regularization="ridge", lam=0.1, feature_names=features
    )
    model2.fit(X_train, y_train)
    save_model(model2, "models/model3/ridge_model3.pkl")
    save_weights_csv(model2, "results/model3/ridge_weights3.csv")

    # Model 3: Lasso Regression (L1)
    model3 = RegressionModel(
        learning_rate=0.001, epochs=5000,
        regularization="lasso", lam=0.01, feature_names=features
    )
    model3.fit(X_train, y_train)
    save_model(model3, "models/model3/lasso_model.pkl")
    save_weights_csv(model3, "results/model3/lasso_weights.csv")

    # Predictions
   # -----------------------------
# Save Metrics for All Models
# -----------------------------
    models = {
    "OLS": model1,
    "Ridge": model2,
    "Lasso": model3
}

    with open("results/model3/test_metrics.txt", "w") as f:
      for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        f.write(f"{name} Regression Metrics (Train):\n")
        f.write(f"MSE: {mse(y_train, y_train_pred):.2f}\n")
        f.write(f"RMSE: {rmse(y_train, y_train_pred):.2f}\n")
        f.write(f"R²: {r2_score(y_train, y_train_pred):.2f}\n\n")

        f.write(f"{name} Regression Metrics (Test):\n")
        f.write(f"MSE: {mse(y_test, y_test_pred):.2f}\n")
        f.write(f"RMSE: {rmse(y_test, y_test_pred):.2f}\n")
        f.write(f"R²: {r2_score(y_test, y_test_pred):.2f}\n\n")
        f.write("="*50 + "\n\n")

    print("✅ Saved metrics for OLS, Ridge, and Lasso in results/test_metrics.txt")
