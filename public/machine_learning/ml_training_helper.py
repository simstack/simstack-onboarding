import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from odmantic import ObjectId
from simstack.core.context import context
from simstack.models import BooleanData
from simstack.models.charts_artifact import AGHeatmapSeriesConfig, create_simple_heatmap_chart, create_simple_scatter_chart
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
        parent_id=ObjectId(task_id) if task_id else None
    )
    await context.db.save(chart)
    return chart


async def save_heatmap_plot(
    data,
    *,
    x_key,
    y_key,
    color_key,
    title,
    task_id,
    x_name,
    y_name,
    color_name,
    color_domain=None,
    color_range=None,
):
    chart = create_simple_heatmap_chart(
        data=data,
        x_key=x_key,
        y_key=y_key,
        color_key=color_key,
        title=title,
        parent_id=ObjectId(task_id) if task_id else None,
    )
    if chart.series and isinstance(chart.series[0], AGHeatmapSeriesConfig):
        chart.series[0].xName = x_name
        chart.series[0].yName = y_name
        chart.series[0].colorName = color_name
        if color_domain is not None:
            chart.series[0].colorDomain = color_domain
        if color_range is not None:
            chart.series[0].colorRange = color_range
    await context.db.save(chart)
    return chart


def _format_interval_label(interval) -> str:
    return f"{interval.left:.3g} to {interval.right:.3g}"


def _build_prediction_density_heatmap_data(y_true, y_pred, *, bins: int = 12):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    combined = np.concatenate([actual, predicted])
    if combined.size == 0 or np.nanmin(combined) == np.nanmax(combined):
        return []

    bin_count = min(bins, max(2, np.unique(combined).size))
    bin_edges = np.linspace(np.nanmin(combined), np.nanmax(combined), bin_count + 1)
    frame = pd.DataFrame(
        {
            "actual_bin": pd.cut(actual, bins=bin_edges, include_lowest=True, duplicates="drop"),
            "predicted_bin": pd.cut(predicted, bins=bin_edges, include_lowest=True, duplicates="drop"),
        }
    ).dropna()
    if frame.empty:
        return []

    grouped = frame.groupby(["actual_bin", "predicted_bin"], observed=True).size().reset_index(name="sample_count")
    return [
        {
            "actual_bin": _format_interval_label(row["actual_bin"]),
            "predicted_bin": _format_interval_label(row["predicted_bin"]),
            "sample_count": int(row["sample_count"]),
        }
        for _, row in grouped.iterrows()
    ]


def _build_property_error_heatmap_data(x_values, y_values, error_values, *, bins: int = 10):
    frame = pd.DataFrame(
        {
            "x_value": np.asarray(x_values, dtype=float).reshape(-1),
            "y_value": np.asarray(y_values, dtype=float).reshape(-1),
            "abs_error": np.asarray(error_values, dtype=float).reshape(-1),
        }
    ).dropna()
    if frame.empty:
        return []

    x_bin_count = min(bins, int(frame["x_value"].nunique()))
    y_bin_count = min(bins, int(frame["y_value"].nunique()))
    if x_bin_count < 2 or y_bin_count < 2:
        return []

    frame["x_bin"] = pd.cut(frame["x_value"], bins=x_bin_count, duplicates="drop")
    frame["y_bin"] = pd.cut(frame["y_value"], bins=y_bin_count, duplicates="drop")
    grouped = (
        frame.dropna(subset=["x_bin", "y_bin"])
        .groupby(["x_bin", "y_bin"], observed=True)["abs_error"]
        .mean()
        .reset_index(name="mean_abs_error")
    )
    return [
        {
            "x_bin": _format_interval_label(row["x_bin"]),
            "y_bin": _format_interval_label(row["y_bin"]),
            "mean_abs_error": float(row["mean_abs_error"]),
        }
        for _, row in grouped.iterrows()
    ]


def _target_label(target: str) -> str:
    return target.replace("_wt_percent", "").replace("_", " ")


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
        df_clean = df[self.features + self.targets].dropna().copy()

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
        performance_rows = []

        for i, target in enumerate(self.targets):
            # Metrics
            y_true_test = self.y_test.iloc[:, i] if len(self.targets) > 1 else self.y_test
            y_true_train = self.y_train.iloc[:, i] if len(self.targets) > 1 else self.y_train
            train_mse = mean_squared_error(y_true_train, y_pred_train[:, i])
            train_r2 = r2_score(y_true_train, y_pred_train[:, i])
            mse = mean_squared_error(y_true_test, y_pred_test[:, i])
            r2 = r2_score(y_true_test, y_pred_test[:, i])
            train_rmse = float(np.sqrt(train_mse))
            test_rmse = float(np.sqrt(mse))
            metrics[f"{target}_mse"] = float(mse)
            metrics[f"{target}_r2"] = float(r2)
            metrics[f"{target}_train_r2"] = float(train_r2)
            metrics[f"{target}_train_rmse"] = train_rmse
            metrics[f"{target}_test_rmse"] = test_rmse
            self.node_runner.info(f"Target {target}: MSE={mse:.6f}, R2={r2:.4f}")
            performance_rows.extend(
                [
                    {"target": _target_label(target), "split": "Train", "r2": float(train_r2), "rmse": train_rmse},
                    {"target": _target_label(target), "split": "Test", "r2": float(r2), "rmse": test_rmse},
                ]
            )

            # Scatter Plots using ChartArtifactModel
            # Training data
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

            density_heatmap_data = _build_prediction_density_heatmap_data(y_true_test, y_pred_test[:, i])
            if density_heatmap_data:
                await save_heatmap_plot(
                    density_heatmap_data,
                    x_key="actual_bin",
                    y_key="predicted_bin",
                    color_key="sample_count",
                    title=f"Actual vs Predicted Density Heatmap: {_target_label(target)}",
                    task_id=self.task_id,
                    x_name="Actual value bin",
                    y_name="Predicted value bin",
                    color_name="Sample count",
                    color_range=["#f7fbff", "#6baed6", "#08306b"],
                )

            if "yield_strength_MPa" in self.X_test.columns and "fracture_strain" in self.X_test.columns:
                error_heatmap_data = _build_property_error_heatmap_data(
                    self.X_test["yield_strength_MPa"],
                    self.X_test["fracture_strain"],
                    np.abs(np.asarray(y_true_test, dtype=float).reshape(-1) - y_pred_test[:, i]),
                )
                if error_heatmap_data:
                    await save_heatmap_plot(
                        error_heatmap_data,
                        x_key="x_bin",
                        y_key="y_bin",
                        color_key="mean_abs_error",
                        title=f"Prediction Error in Property Space: {_target_label(target)}",
                        task_id=self.task_id,
                        x_name="Yield strength bin (MPa)",
                        y_name="Fracture strain bin",
                        color_name="Mean absolute error",
                        color_range=["#fff5eb", "#fdae6b", "#a63603"],
                    )

        # Total metrics
        total_r2 = float(r2_score(self.y_test, y_pred_test, multioutput='uniform_average'))
        self.node_runner.info(f"Average R2 Score: {total_r2:.4f}")
        metrics["average_r2"] = total_r2

        if performance_rows:
            r2_heatmap_data = [
                {"target": row["target"], "split": row["split"], "r2": row["r2"]}
                for row in performance_rows
            ]
            await save_heatmap_plot(
                r2_heatmap_data,
                x_key="target",
                y_key="split",
                color_key="r2",
                title="Prediction R2 Heatmap",
                task_id=self.task_id,
                x_name="Impurity target",
                y_name="Dataset split",
                color_name="R2 score",
                color_domain=[-1.0, 1.0],
                color_range=["#b2182b", "#f7f7f7", "#2166ac"],
            )

            rmse_heatmap_data = [
                {"target": row["target"], "split": row["split"], "rmse": row["rmse"]}
                for row in performance_rows
            ]
            max_rmse = max((row["rmse"] for row in performance_rows), default=0.0)
            await save_heatmap_plot(
                rmse_heatmap_data,
                x_key="target",
                y_key="split",
                color_key="rmse",
                title="Prediction RMSE Heatmap",
                task_id=self.task_id,
                x_name="Impurity target",
                y_name="Dataset split",
                color_name="RMSE",
                color_domain=[0.0, max_rmse] if max_rmse > 0 else None,
                color_range=["#ffffcc", "#fd8d3c", "#800026"],
            )

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
