from simstack.models import BooleanData, IntData
from sklearn.ensemble import RandomForestRegressor
from simstack.core.context import context
from simstack.models.pandas_model import PandasModel

from models.element_selector import ElementSelector
from simstack.core.node import node
from simstack.models import Parameters

from public.machine_learning.ml_training_helper import RegressionAnalysis


@node(parameters=Parameters(force_rerun=True))
async def train_impurity_regressor_full(dataset: PandasModel, n_estimators: IntData,
                                   max_depth: IntData,
                                   r_seed: IntData, element_selector: ElementSelector,
                                   use_scaling: BooleanData, use_engineered_features: BooleanData, **kwargs):
    """
    Load the training_data dataset and train a RandomForestRegressor 
    to predict impurity concentrations (C, Mn, P, S) based on extracted features.
    """
    node_runner = kwargs.get("node_runner")
    node_runner.info(f"Training Impurity Regressor with {n_estimators.value} estimators, max_depth {max_depth.value} and seed {r_seed.value}")
    assert max_depth.value > 0, "max_depth must be greater than 0"
    assert n_estimators.value > 0, "n_estimators must be greater than 0"
    assert r_seed.value > 0, "r_seed must be greater than 0"
    assert max_depth.value < 20, "max_depth must be less than 20"
    assert n_estimators.value < 200, "n_estimators must be less than 200"
    # 3. Setup Regression Analysis
    # kwargs can include use_scaling and use_engineered_features flags
    analysis = RegressionAnalysis(dataset, element_selector, use_scaling_input=use_scaling, use_engineered_features_input=use_engineered_features, **kwargs)
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


async def train_impurity_regressor(dataset: PandasModel, selector: ElementSelector, **kwargs):
    """
    Compatibility wrapper or simplified version.
    """
    n_estimators = IntData(value=10)
    max_depth = IntData(value=5)
    r_seed = IntData(value=42)
    use_scaling = kwargs.get("use_scaling", BooleanData(value=False))
    use_engineered_features = kwargs.get("use_engineered_features", BooleanData(value=False))
    return await train_impurity_regressor_full(
        dataset, n_estimators, max_depth, r_seed, selector, use_scaling, use_engineered_features, **kwargs
    )


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

