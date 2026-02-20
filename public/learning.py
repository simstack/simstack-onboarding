import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from simstack.core.node import node
from simstack.models.array_storage import ArrayStorage

@node
def load_data(storage: ArrayStorage):
    """
    Load data from an ArrayStorage object.
    """
    data = storage.get_array()
    return data

@node
def split_data(storage: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets.
    Assumes the last column is the target.
    """
    node_runner =
    assert data.ndim == 2, "Data must be two-dimensional."
    assert data.shape[1] > 1, "Data must have at least two columns."
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

@node
def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@node
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate the model on the test set.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"mse": mse, "r2": r2}

@node
def log_metrics(metrics: dict):
    """
    Log the evaluation metrics.
    """
    print(f"Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    return metrics
