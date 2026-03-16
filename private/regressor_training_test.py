from simstack.core.context import context
from simstack.models import BooleanData
from simstack.models.pandas_model import PandasModel

from public.machine_learning.train_impurity_model import train_impurity_regressor


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
    await train_impurity_regressor(dataset, single_selector, node_runner=MockNodeRunner(),
                                   use_scaling=BooleanData(value=False),
                                   use_engineered_features=BooleanData(value=False))

    print("\n--- Training Simplified Model (No Scaling, No Engineering) ---")
    await train_impurity_regressor(dataset, selector, node_runner=MockNodeRunner(),
                                   use_scaling=BooleanData(value=False),
                                   use_engineered_features=BooleanData(value=False))

    print("\n--- Training Advanced Model (Scaling + Engineering) ---")
    await train_impurity_regressor(dataset, selector, node_runner=MockNodeRunner(), use_scaling=BooleanData(value=True),
                                   use_engineered_features=BooleanData(value=True))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

