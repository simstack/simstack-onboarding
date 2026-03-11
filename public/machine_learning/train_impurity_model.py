import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import os
from odmantic import ObjectId
from simstack.models import StringData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from simstack.core.context import context
from simstack.models.pandas_model import PandasModel
from models.regression_result import RegressionResult
from simstack.models.charts_artifact import create_simple_scatter_chart
from simstack.core.node import node

async def save_scatter_plot(y_true, y_pred, title, x_label, y_label, task_id, **kwargs):
    """
    Helper to create and save a scatter plot using ChartArtifactModel.
    """
    # Convert pandas series/dataframe to list
    if hasattr(y_true, 'values'):
        y_true_list = y_true.values.tolist()
    elif hasattr(y_true, 'tolist'):
        y_true_list = y_true.tolist()
    else:
        y_true_list = list(y_true)

    if hasattr(y_pred, 'tolist'):
        y_pred_list = y_pred.tolist()
    else:
        y_pred_list = list(y_pred)

    data = []
    for true_val, pred_val in zip(y_true_list, y_pred_list):
        # Handle cases where true_val might be a single-element list from a 1-column DF
        val = true_val[0] if isinstance(true_val, (list, np.ndarray)) and len(true_val) == 1 else true_val
        pred = pred_val[0] if isinstance(pred_val, (list, np.ndarray)) and len(pred_val) == 1 else pred_val
        data.append({x_label: float(val), y_label: float(pred)})
    
    chart = create_simple_scatter_chart(
        data=data,
        x_key=x_label,
        y_key=y_label,
        title=title,
        parent_id=ObjectId(task_id) if task_id else None
    )
    await context.db.save(chart)
    return chart


class RegressionAnalysis:
    def __init__(self, dataset_name: str, node_runner: Any, task_id: str | None = None, targets: list[str] | None = None):
        self.dataset_name = dataset_name
        self.node_runner = node_runner
        self.task_id = task_id
        self.targets = targets or ["C_wt_percent", "Mn_wt_percent", "P_wt_percent", "S_wt_percent"]
        self.features = [
            "youngs_modulus_MPa",
            "yield_strength_MPa",
            "ultimate_strength_MPa",
            "fracture_stress_MPa",
            "fracture_strain"
        ]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    async def make_model_data(self):
        # 1. Load data
        dataset = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": self.dataset_name}})
        if dataset is None:
            raise ValueError(f"Dataset matching '{self.dataset_name}' not found.")

        df = dataset.table

        # 2. Define Features and Targets
        # Drop rows with NaN if any (though synthetic data should be clean)
        df_clean = df[self.features + self.targets].dropna()

        X = df_clean[self.features]
        y = df_clean[self.targets]

        # 3. Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    async def model_analysis(self, model: RandomForestRegressor) -> RegressionResult:
        self.model = model
        self.node_runner.info(f"Training RandomForestRegressor on {len(self.X_train)} samples for targets: {self.targets}...")
        self.model.fit(self.X_train, self.y_train)

        # 5. Evaluate and Plot
        y_pred_train = self.model.predict(self.X_train)
        y_pred_test = self.model.predict(self.X_test)

        # Handle 1D predictions if only one target
        if len(self.targets) == 1:
            y_pred_train = y_pred_train.reshape(-1, 1)
            y_pred_test = y_pred_test.reshape(-1, 1)

        metrics = {}

        for i, target in enumerate(self.targets):
            # Metrics
            y_true_test = self.y_test.iloc[:, i] if len(self.targets) > 1 else self.y_test
            mse = mean_squared_error(y_true_test, y_pred_test[:, i])
            r2 = r2_score(y_true_test, y_pred_test[:, i])
            metrics[f"{target}_mse"] = float(mse)
            metrics[f"{target}_r2"] = float(r2)
            self.node_runner.info(f"Target {target}: MSE={mse:.6f}, R2={r2:.4f}")

            # Scatter Plots using ChartArtifactModel
            # Training data
            y_true_train = self.y_train.iloc[:, i] if len(self.targets) > 1 else self.y_train
            await save_scatter_plot(
                y_true_train, y_pred_train[:, i],
                title=f"Train: {target}",
                x_label="Expected", y_label="Predicted",
                task_id=self.task_id
            )

            # Test data
            await save_scatter_plot(
                y_true_test, y_pred_test[:, i],
                title=f"Test: {target}",
                x_label="Expected", y_label="Predicted",
                task_id=self.task_id
            )

        # Total metrics
        total_r2 = float(r2_score(self.y_test, y_pred_test, multioutput='uniform_average'))
        self.node_runner.info(f"Average R2 Score: {total_r2:.4f}")
        metrics["average_r2"] = total_r2

        # 6. Save results
        # Save model to local file
        target_suffix = "_".join(self.targets)
        if len(target_suffix) > 50:
            target_suffix = "multi_impurity"
        model_filename = Path.cwd() / f"regressor_model_{target_suffix}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(self.model, f)
        self.node_runner.info(f"Model saved locally to {model_filename}")

        # Save metrics to RegressionResult
        results = RegressionResult(
            field_name=f"impurity_prediction_metrics_{target_suffix}",
            metrics=metrics
        )

        await context.db.save(results)
        self.node_runner.info("Metrics saved to database as RegressionResult.")
        return results


@node
async def train_impurity_regressor(dataset_name_model: StringData, **kwargs):
    """
    Load the training_data dataset and train a RandomForestRegressor 
    to predict impurity concentrations (C, Mn, P, S) based on extracted features.
    """
    node_runner = kwargs.get("node_runner")
    task_id = kwargs.get("task_id")
    dataset_name = dataset_name_model.value

    analysis = RegressionAnalysis(dataset_name, node_runner, task_id)
    await analysis.make_model_data()

    # 4. Train Model
    # RandomForestRegressor supports multi-output out of the box.
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    results = await analysis.model_analysis(model)

    # Return metrics dictionary instead of RegressionResult model to avoid Simstack result processing error
    node_runner.result = results
    return node_runner.succeed()


@node
async def train_sulfur_regressor(dataset_name_model: StringData, **kwargs):
    """
    Load the training_data dataset and train a RandomForestRegressor 
    to predict ONLY Sulfur concentration based on extracted features.
    """
    node_runner = kwargs.get("node_runner")
    task_id = kwargs.get("task_id")
    dataset_name = dataset_name_model.value

    analysis = RegressionAnalysis(dataset_name, node_runner, task_id, targets=["S_wt_percent"])
    await analysis.make_model_data()

    # 4. Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    results = await analysis.model_analysis(model)

    # Return metrics dictionary instead of RegressionResult model to avoid Simstack result processing error
    node_runner.result = results
    return node_runner.succeed()


if __name__ == "__main__":
    import asyncio
    async def run_standalone():
        await context.initialize()
        # Mock node_runner for local execution
        class MockNodeRunner:
            def info(self, msg): print(f"INFO: {msg}")
            def succeed(self): return "Success"
            result = None
        
        # Bypass @node decorator for local verification due to TaskStatus.FAILED persistence in DB
        task_id = str(ObjectId())
        
        print("\n--- Training All Impurities ---")
        analysis_all = RegressionAnalysis("training_data", MockNodeRunner(), task_id)
        await analysis_all.make_model_data()
        model_all = RandomForestRegressor(n_estimators=100, random_state=42)
        await analysis_all.model_analysis(model_all)
        
        print("\n--- Training Sulfur Only ---")
        analysis_s = RegressionAnalysis("training_data", MockNodeRunner(), task_id, targets=["S_wt_percent"])
        await analysis_s.make_model_data()
        model_s = RandomForestRegressor(n_estimators=100, random_state=42)
        await analysis_s.model_analysis(model_s)
        
    asyncio.run(run_standalone())
