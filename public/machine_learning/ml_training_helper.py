import pickle
from pathlib import Path
from typing import Any

import numpy as np
from odmantic import ObjectId
from simstack.core.context import context
from simstack.models import BooleanData
from simstack.models.charts_artifact import create_simple_scatter_chart
from simstack.models.pandas_model import PandasModel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.element_selector import ElementSelector
from models.regression_result import RegressionResult


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
    )
    await context.db.save(chart)
    return chart

default_element_selector = ElementSelector(use_C=True, use_Mn=True, use_P=True, use_S=True)

class RegressionAnalysis:
    def __init__(self, dataset: PandasModel | str, target_selector: ElementSelector | list[str]  = default_element_selector,
                 use_scaling_input: BooleanData = BooleanData(value=False),
                 use_engineered_features_input: BooleanData = BooleanData(value=False), **kwargs):
        self.node_runner = kwargs.get("node_runner")
        self.task_id = kwargs.get("task_id")
        self.dataset = dataset
        self.targets = []
        if isinstance(target_selector, list):
            self.targets = target_selector
        else:
            if target_selector.use_C:
                self.targets.append("C_wt_percent")
            if target_selector.use_Mn:
                self.targets.append("Mn_wt_percent")
            if target_selector.use_P:
                self.targets.append("P_wt_percent")
            if target_selector.use_S:
                self.targets.append("S_wt_percent")
        if len(self.targets) == 0:
            raise ValueError("At least one target must be selected.")
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

        # Onboarding options
        self.use_scaling = use_scaling_input.value
        self.use_engineered_features = use_engineered_features_input.value

    async def make_model_data(self):
        # 1. Load data
        if isinstance(self.dataset, str):
            dataset_model = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": self.dataset}})
            if dataset_model is None:
                raise ValueError(f"Dataset matching '{self.dataset}' not found.")
            df = dataset_model.table
        else:
            df = self.dataset.table

        # 2. Define Features and Targets
        # Drop rows with NaN if any (though synthetic data should be clean)
        df_clean = df[self.features + self.targets].dropna()

        X_cols = self.features.copy()
        if self.use_engineered_features:
            # Add engineering features
            # Since yield and UTS depend on sqrt(C), maybe the model needs more help
            # But for P and S, the relationship is linear with strains.
            # Let's check if the ratio of strengths or strains helps.
            df_clean["strength_ratio"] = df_clean["ultimate_strength_MPa"] / df_clean["yield_strength_MPa"]
            df_clean["strain_diff"] = df_clean["fracture_strain"] - (df_clean["ultimate_strength_MPa"] / df_clean["youngs_modulus_MPa"])
            X_cols += ["strength_ratio", "strain_diff"]

        X = df_clean[X_cols]
        y = df_clean[self.targets]

        # 3. Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    async def model_analysis(self, model: Any) -> RegressionResult:
        self.model = model
        self.node_runner.info(f"Training {type(model).__name__} on {len(self.X_train)} samples for targets: {self.targets}...")

        if self.use_scaling:
            # Scaling targets to ensure equal weight in MultiOutput models
            self.target_scaler = StandardScaler()
            y_train_fit = self.target_scaler.fit_transform(self.y_train)
        else:
            y_train_fit = self.y_train

        # Reshape to 1D if single target to avoid sklearn warning
        if len(self.targets) == 1:
            if hasattr(y_train_fit, "values"):
                y_train_fit = y_train_fit.values.ravel()
            else:
                y_train_fit = np.array(y_train_fit).ravel()

        self.model.fit(self.X_train, y_train_fit)

        # 5. Evaluate and Plot
        y_pred_train_raw = self.model.predict(self.X_train)
        y_pred_test_raw = self.model.predict(self.X_test)

        # Reshape if necessary
        if len(self.targets) == 1:
            y_pred_train_raw = y_pred_train_raw.reshape(-1, 1)
            y_pred_test_raw = y_pred_test_raw.reshape(-1, 1)

        if self.use_scaling:
            y_pred_train = self.target_scaler.inverse_transform(y_pred_train_raw)
            y_pred_test = self.target_scaler.inverse_transform(y_pred_test_raw)
        else:
            y_pred_train = y_pred_train_raw
            y_pred_test = y_pred_test_raw

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
            setattr(self.node_runner,f"train.{target}", await save_scatter_plot(
                y_true_train, y_pred_train[:, i],
                title=f"Train: {target}",
                x_label="Expected", y_label="Predicted",
                task_id=self.task_id
            ))

            # Test data
            setattr(self.node_runner, f"test.{target}",
                    await save_scatter_plot(
                y_true_test, y_pred_test[:, i],
                title=f"Test: {target}",
                x_label="Expected", y_label="Predicted",
                task_id=self.task_id
            ))

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
