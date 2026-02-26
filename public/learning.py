import asyncio

import numpy as np
from docutils.nodes import field_name
from pandas.core.interchange.dataframe_protocol import DataFrame
from simstack.core.context import context
from simstack.models import StringData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from simstack.core.node import node
from simstack.models.array_storage import ArrayStorage
from simstack.models.artifact_models import ArtifactModel

from public.models.df_model import DataFrameModel


@node
async def load_data(name: StringData,**kwargs) -> DataFrame:
    """
    Load data from an ArrayStorage object.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info(f"loading data from {name.value}")
    storage = await context.db.engine.find_one(DataFrameModel, {"field_name": name.value})
    if storage is None:
        raise ValueError(f"Data with field_name {name.value} not found in database.")
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
    
    # Return as an ArtifactModel
    return ArtifactModel(name="random_forest_model", data={"model": model})

@node
def evaluate_model(model: ArtifactModel, X_test: ArrayStorage, y_test: ArrayStorage):
    """
    Evaluate the model on the test set.
    """
    rf_model = model.data["model"]
    predictions = rf_model.predict(X_test.get_array())
    mse = mean_squared_error(y_test.get_array(), predictions)
    r2 = r2_score(y_test.get_array(), predictions)
    
    return ArtifactModel(name="evaluation_metrics", data={"mse": mse, "r2": r2})

@node
def log_metrics(metrics: ArtifactModel):
    """
    Log the evaluation metrics.
    """
    print(f"Model Evaluation Metrics:")
    for metric, value in metrics.data.items():
        print(f"  {metric}: {value:.4f}")
    return metrics

async def main():
    await context.initialize()
    result = await load_data(StringData(field_name="input", value="housing_dataset"))

if __name__ == "__main__":
    asyncio.run(main())