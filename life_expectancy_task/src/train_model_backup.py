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
# Regression Model
# -----------------------------
class RegressionModel:
    def __init__(self, method="gradient_descent", learning_rate=0.01, epochs=1000,
                 regularization="lasso", lam=0.1, degree=1, feature_names=None):
        self.method = method
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
        # Add bias/intercept
        X = np.c_[np.ones((X.shape[0], 1)), X]
        X = self._expand_polynomial(X)

        if self.method == "closed_form":
            if self.regularization is None:
                # Ordinary Least Squares
                self.weights = np.linalg.inv(X.T @ X) @ (X.T @ y)
            elif self.regularization == "ridge":
                I = np.eye(X.shape[1])
                I[0, 0] = 0  # don't regularize bias
                self.weights = np.linalg.inv(X.T @ X + self.lam * I) @ (X.T @ y)
            elif self.regularization == "lasso":
                raise ValueError("Closed-form Lasso not supported. Use gradient descent.")
            else:
                raise ValueError("Unknown regularization type.")

        elif self.method == "gradient_descent":
            self.weights = np.zeros(X.shape[1])
            for _ in range(self.epochs):
                y_pred = X @ self.weights
                error = y_pred - y
                grad = (X.T @ error) / len(y)

                if self.regularization == "ridge":
                    grad += self.lam * self.weights
                    grad[0] -= self.lam * self.weights[0]  # don't regularize bias
                elif self.regularization == "lasso":
                    grad += self.lam * np.sign(self.weights)
                    grad[0] -= self.lam * np.sign(self.weights[0])  # don't regularize bias

                self.weights -= self.learning_rate * grad
        else:
            raise ValueError("Unknown method for training")

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        X = self._expand_polynomial(X)
        return X @ self.weights


# -----------------------------
# Save/Load Model
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
        " thinness  1-19 years", "Income composition of resources", "Schooling"
    ]

    # features = ["Health_Index", "Economic_Index", "Disease_Index"]
    X, y = load_data(
        "/Users/anandagarwal/life_expectancy_task/data/Life Expectancy.csv",
        features=features,
        target="Life expectancy "
    )

    # Train-test split (80% train, 20% test)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Model 1: Linear Regression (Closed-form OLS)
    model1 = RegressionModel(method="gradient_descent", regularization="lasso", feature_names=features)
    model1.fit(X_train, y_train)
    save_model(model1, "models/regression_model1.pkl")
    save_weights_csv(model1, "results/lasso_weights.csv")

    # # Model 2: Lasso Regression (Gradient Descent)
    # model2 = RegressionModel(method="gradient_descent", regularization="lasso",
    #                          learning_rate=0.001, epochs=5000, lam=0.01, feature_names=features)
    # model2.fit(X_train, y_train)
    # save_model(model2, "models/regression_lasso.pkl")
    # save_weights_csv(model2, "results/lasso_weights.csv")

    # Predictions
    y_train_pred = model1.predict(X_train)
    y_test_pred = model1.predict(X_test)

    # Save metrics
    with open("results/train_metrics.txt", "w") as f:
        f.write("Regression Metrics (Train):\n")
        f.write(f"Mean Squared Error (MSE): {mse(y_train, y_train_pred):.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse(y_train, y_train_pred):.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_score(y_train, y_train_pred):.2f}\n\n")

        f.write("Regression Metrics (Test):\n")
        f.write(f"Mean Squared Error (MSE): {mse(y_test, y_test_pred):.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse(y_test, y_test_pred):.2f}\n")
        f.write(f"R-squared (R²) Score: {r2_score(y_test, y_test_pred):.2f}\n")

    # Save test predictions
    np.savetxt("results/train_predictions.csv", y_test_pred, delimiter=",", fmt="%.4f")
