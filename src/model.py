import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing import FEATURES

def evaluate(y_true, y_pred, split_name="Test"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n── {split_name} Results ──")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

class BaselineModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        print("Training baseline (Linear Regression)...")
        self.model.fit(X_train, y_train)
        print("Done.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, split_name="Test"):
        preds = self.predict(X_test)
        return evaluate(y_test, preds, split_name)

    def save(self, path="models/baseline.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path="models/baseline.pkl"):
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj


class ImprovedModel:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )

    def train(self, X_train, y_train):
        print("Training improved model (Random Forest)...")
        self.model.fit(X_train, y_train)
        print("Done.")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, split_name="Test"):
        preds = self.predict(X_test)
        return evaluate(y_test, preds, split_name)

    def feature_importance(self):
        importance = pd.Series(
            self.model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)
        print("\nFeature Importance:")
        print(importance.round(4))
        return importance

    def save(self, path="models/improved.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path="models/improved.pkl"):
        obj = cls()
        with open(path, "rb") as f:
            obj.model = pickle.load(f)
        return obj
