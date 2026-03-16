from simstack.models import BooleanData
from sklearn.ensemble import RandomForestRegressor
from simstack.core.context import context
from simstack.models.pandas_model import PandasModel

from models.element_selector import ElementSelector
from simstack.core.node import node
from simstack.models import Parameters

from public.machine_learning.ml_training_helper import RegressionAnalysis

@node(parameters=Parameters(force_rerun=True))
async def train_impurity_regressor(dataset: PandasModel, **kwargs):
    """
    Load the training_data dataset and train a RandomForestRegressor 
    to predict impurity concentrations (C, Mn, P, S) based on extracted features.
    """
    node_runner = kwargs.get("node_runner")
    
    # 3. Setup Regression Analysis
    analysis = RegressionAnalysis(dataset, **kwargs)
    await analysis.make_model_data()

    # 4. Train Model
    # Using RandomForestRegressor for onboarding simplicity.
    # It handles multi-output targets automatically.
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        
    results = await analysis.model_analysis(model)

    # Return metrics dictionary instead of RegressionResult model to avoid Simstack result processing error
    node_runner.result = results
    return node_runner.succeed()

