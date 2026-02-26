import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from simstack.core.node import node
from simstack.models.array_storage import ArrayStorage
from simstack.models import simstack_model
from odmantic import Model

@simstack_model
class ScikitModel(Model):
    def __init__(self, model):
        super().__init__()
        self._model = model
    
    def predict(self, X):
        return self._model.predict(X)
    
    def fit(self, X, y):
        return self._model.fit(X, y)

@node
def load_data(storage: ArrayStorage) -> ArrayStorage:
    """
    Load data from an ArrayStorage object.
    """
    return storage

@node
def split_data(storage: ArrayStorage, test_size: float = 0.2, random_state: int = 42):
    """
    Split the data into training and testing sets.
    Assumes the last column is the target.
    """
    data = storage.get_array()
    assert data.ndim == 2, "Data must be two-dimensional."
    assert data.shape[1] > 1, "Data must have at least two columns."
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Wrap in ArrayStorage as required by simstack-models
    X_train_storage = ArrayStorage(name="X_train")
    X_train_storage.set_array(X_train)
    X_test_storage = ArrayStorage(name="X_test")
    X_test_storage.set_array(X_test)
    y_train_storage = ArrayStorage(name="y_train")
    y_train_storage.set_array(y_train)
    y_test_storage = ArrayStorage(name="y_test")
    y_test_storage.set_array(y_test)
    
    return X_train_storage, X_test_storage, y_train_storage, y_test_storage

@node
def train_model(X_train: ArrayStorage, y_train: ArrayStorage):
    """
    Train a Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.get_array(), y_train.get_array())
    return model

@node
def evaluate_model(model, X_test: ArrayStorage, y_test: ArrayStorage):
    """
    Evaluate the model on the test set.
    """
    predictions = model.predict(X_test.get_array())
    mse = mean_squared_error(y_test.get_array(), predictions)
    r2 = r2_score(y_test.get_array(), predictions)
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
