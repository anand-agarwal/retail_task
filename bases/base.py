import numpy as np
import pandas as pd
from pathlib import Path
import os
import pickle
import csv


class BaseModel:
    folder_name = None

    def __init__(self):
        self.df: pd.DataFrame = None
        self.read_csv()

    @staticmethod
    def _get_path(folder_name, name='train_data.csv'):
        full_path = os.path.realpath(__file__)
        p = Path(os.path.dirname(full_path))
        return p.parent / folder_name / "data" / name

    def one_hot_encode(self, col, prefix=None):
        if prefix is None: prefix = col
        oh = pd.get_dummies(self.df[col], prefix=prefix, drop_first=True, dtype="uint8")
        self.df = self.df.drop(columns=[col]).join(oh)

    def double_encode(self, col):
        self.df[col] = self.df[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0}).astype('Int64')

    def read_csv(self):
        p = self._get_path(self.folder_name)

        if not (p.parent / 'df.pkl').exists():
            self.df = pd.read_csv(p)
            with open(p.parent / 'df.pkl', 'wb') as f:
                pickle.dump(self.df, f)

        with open(p.parent / 'df.pkl', 'rb') as f:
            self.df = pickle.load(f)

    def preprocess(self):
        pass

    def standardize(self, target=None):

        df = self.df.copy()
        num = df.select_dtypes(include=[np.number]).columns

        # drop target if numeric
        if target is not None and target in num:
            num = num.drop(target)

        # detect 0/1 columns (one-hots)
        is_ohe = df[num].apply(lambda s: s.dropna().isin([0, 1]).all())
        ohe_cols = is_ohe[is_ohe].index

        cont = num.difference(ohe_cols)  # continuous columns to scale

        mu = df[cont].mean()
        sig = df[cont].std(ddof=0).replace(0, 1.0)

        df[cont] = (df[cont] - mu) / sig
        self.df = df

    def remove_nan(self):
        pass


    def extract_x_y(self, y_col):
        cols = [c for c in self.df.columns if c != y_col]
        return self.df[cols].values, self.df[y_col].values

    def train_test_split(self, X, y):
        # Train-test split (80/20)
        np.random.seed(42)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        split = int(0.8 * len(X))
        train_idx, test_idx = indices[:split], indices[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        return X_train, y_train, X_test, y_test


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

    def _soft_threshold(self, z, t):
        return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)

    def fit(self, X, y):
        # ensure float64 tensors
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n = X.shape[0]

        # 1) expand ONLY real features, not the bias
        X_feat = self._expand_polynomial(X)  # make sure this does NOT add a bias
        # 2) standardize AFTER expansion (exclude bias later)
        mu = X_feat.mean(axis=0)
        sigma = X_feat.std(axis=0)
        sigma[sigma == 0] = 1.0
        X_feat = (X_feat - mu) / sigma

        # 3) now add bias as the first column
        Xb = np.c_[np.ones((n, 1)), X_feat]

        # 4) init
        self.weights = np.zeros(Xb.shape[1], dtype=np.float64)
        lr = getattr(self, "learning_rate", 1e-3)
        lam = getattr(self, "lam", 0.0)

        # optional: gradient clipping threshold
        clip = 1e3

        for _ in range(self.epochs):
            print(_, "epochs done")
            y_pred = Xb @ self.weights
            error = y_pred - y
            grad = (Xb.T @ error) / n  # MSE gradient

            # regularization (don’t touch bias at index 0)
            if self.regularization == "ridge":
                grad[1:] += lam * self.weights[1:]
            elif self.regularization == "lasso":
                # use proximal step (ISTA) for stability instead of raw sign()
                # do the gradient step on w first (no reg on bias)
                w0 = self.weights.copy()
                w_tmp = w0 - lr * grad
                w_tmp[1:] = self._soft_threshold(w_tmp[1:], lr * lam)
                w_tmp[0] = w0[0] - lr * grad[0]  # bias plain step
                self.weights = w_tmp
                # finite check
                if not np.all(np.isfinite(self.weights)):
                    raise FloatingPointError("Weights became non-finite; reduce lr or scale features.")
                continue  # already updated, skip the line below

            # gradient clipping (helps with outliers / early iters)
            np.clip(grad, -clip, clip, out=grad)

            # update
            self.weights -= lr * grad

            # safety: bail if things explode
            if not np.all(np.isfinite(self.weights)):
                raise FloatingPointError("Weights became non-finite; reduce lr or scale features.")

        # store scaling params for predict()
        self._mu_ = mu
        self._sigma_ = sigma

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_feat = self._expand_polynomial(X)
        X_feat = (X_feat - self._mu_) / self._sigma_
        Xb = np.c_[np.ones((X_feat.shape[0], 1)), X_feat]
        return Xb @ self.weights

    def save_weights_csv(self, path):
        """Save model weights + corresponding feature names to CSV."""
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(self.weights) - 1)]
        else:
            feature_names = ["bias"] + self.feature_names

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Feature", "Weight"])
            for name, w in zip(feature_names, self.weights):
                writer.writerow([name, w])

    def save_model(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class TrainModels:

    def __init__(self, model_info: BaseModel, target_col):
        self.model_info = model_info
        self.target_col = target_col

    def train_and_save(self, parent_folder, model_folder):
        X, y = self.model_info.extract_x_y(self.target_col)
        cols = [c for c in self.model_info.df.columns if c != self.target_col]
        X_train, y_train, X_test, y_test = self.model_info.train_test_split(X, y)
        full_path = os.path.realpath(__file__)
        p = Path(os.path.dirname(full_path)).parent
        model_p = p / parent_folder / "models" / model_folder
        result_p = p / parent_folder / "results" / model_folder
        model_p.mkdir(parents=True, exist_ok=True)
        result_p.mkdir(parents=True, exist_ok=True)

        model1 = RegressionModel(
            learning_rate=0.001, epochs=750,
            regularization=None, feature_names=cols
        )
        # model1.fit(X_train, y_train)
        # model1.save_model(model_p / "ols_model.pkl")

        model2 = RegressionModel(
            learning_rate=0.001, epochs=750,
            regularization="ridge", lam=0.1, feature_names=cols
        )
        # model2.fit(X_train, y_train)
        # model2.save_model(model_p / "ridge_model.pkl")
        # model2.save_weights_csv(result_p / "ridge_weights.csv")

        model3 = RegressionModel(
            learning_rate=0.001, epochs=5000,
            regularization="lasso", lam=0.01, feature_names=cols
        )
        model3.fit(X_train, y_train)
        model3.save_model(model_p / "lasso_model.pkl")
        model3.save_weights_csv(result_p / "lasso_weights.csv")

        models = {
            # "OLS": model1,
            # "Ridge": model2,
            "Lasso": model3
        }

        print('writing this model now')
        with open(result_p / "test_metrics.txt", "w") as f:
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
                f.write("=" * 50 + "\n\n")


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)






