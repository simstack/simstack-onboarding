from simstack.models import BooleanData, IntData
from sklearn.ensemble import RandomForestRegressor
from simstack.core.context import context
from simstack.models.pandas_model import PandasModel

from models.element_selector import ElementSelector
from simstack.core.node import node
from simstack.models import Parameters

from public.machine_learning.ml_training_helper import RegressionAnalysis

@node(parameters=Parameters(force_rerun=True))
async def train_impurity_regressor(dataset: PandasModel, n_estimators: IntData,
                                   max_depth: IntData,
                                   r_seed: IntData, **kwargs):
    """
    Load the training_data dataset and train a RandomForestRegressor 
    to predict impurity concentrations (C, Mn, P, S) based on extracted features.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info(f"Training Impurity Regressor with {n_estimators.value} estimators, max_depth {max_depth.value} and seed {r_seed.value}")
    assert max_depth.value > 0, "max_depth must be greater than 0"
    assert n_estimators.value > 0, "n_estimators must be greater than 0"
    assert r_seed.value > 0, "r_seed must be greater than 0"
    assert max_depth.value < 20, "r_seed must be less than 20"
    assert n_estimators.value < 200, "n_estimators must be less than 100"

    # 3. Setup Regression Analysis
    analysis = RegressionAnalysis(dataset, **kwargs)
    await analysis.make_model_data()

    # 4. Train Model
    # Using RandomForestRegressor for onboarding simplicity.
    # It handles multi-output targets automatically.
    model = RandomForestRegressor(n_estimators=n_estimators.value, max_depth=max_depth.value,
                                  random_state=r_seed.value)
        
    results = await analysis.model_analysis(model)

    # Return metrics dictionary instead of RegressionResult model to avoid Simstack result processing error
    node_runner.result = results
    return node_runner.succeed()

