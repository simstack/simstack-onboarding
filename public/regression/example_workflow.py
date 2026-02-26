import numpy as np
from simstack.models.array_storage import ArrayStorage
from public.models.df_model import DataFrameModel
from public.regression import load_data, split_data, train_model, evaluate_model, log_metrics

def run_regression_workflow():
    # 1. Setup data in DataFrameModel
    storage = DataFrameModel(field_name="housing_data")
    
    # Mock data: 100 samples, 5 features, 1 target
    import pandas as pd
    df = pd.DataFrame(np.random.rand(100, 6))
    storage.store_df(df)
    
    # 2. Define the workflow by calling nodes
    # In simstack, calling a @node decorated function creates a Node and manages its execution.
    
    # split_data
    X_train, X_test, y_train, y_test = split_data(storage=storage)
    
    # train_model
    model = train_model(X_train=X_train, y_train=y_train)
    
    # evaluate_model
    metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)
    
    # log_metrics
    log_metrics(metrics=metrics)

if __name__ == "__main__":
    # Note: Running this requires a full simstack environment (DB, Engine, etc.)
    # This is a conceptual example showing how the nodes are used.
    print("Regression Workflow nodes defined and ready.")
    # run_regression_workflow() # Uncomment to run if environment is fully setup
