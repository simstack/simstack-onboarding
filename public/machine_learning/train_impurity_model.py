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


async def main():
    await context.initialize()
    from models.element_selector import ElementSelector
    
    # Mock NodeRunner
    class MockNodeRunner:
        def info(self, msg): print(f"INFO: {msg}")
        def succeed(self): print("SUCCESS")
        def fail(self, msg): print(f"FAIL: {msg}")

    # Load dataset
    dataset = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": "training_data"}})
    if not dataset:
        print("Dataset not found. Please run make_training_data first.")
        return

    selector = ElementSelector(use_C=True, use_Mn=True, use_P=True, use_S=True)
    
    print("\n--- Training Single Target Model (S_wt_percent) ---")
    single_selector = ElementSelector(use_C=False, use_Mn=False, use_P=False, use_S=True)
    await train_impurity_regressor(dataset, single_selector, node_runner=MockNodeRunner(), use_scaling=BooleanData(value=False), use_engineered_features=BooleanData(value=False))

    print("\n--- Training Simplified Model (No Scaling, No Engineering) ---")
    await train_impurity_regressor(dataset, selector, node_runner=MockNodeRunner(), use_scaling=BooleanData(value=False), use_engineered_features=BooleanData(value=False))

    print("\n--- Training Advanced Model (Scaling + Engineering) ---")
    await train_impurity_regressor(dataset, selector, node_runner=MockNodeRunner(), use_scaling=BooleanData(value=True), use_engineered_features=BooleanData(value=True))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

