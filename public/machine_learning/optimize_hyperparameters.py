from simstack.core.node import node
from simstack.models import IntData, Parameters
from simstack.models.pandas_model import PandasModel
from simstack.models.charts_artifact import ChartArtifactModel, AGChartTitleConfig, AGScatterSeriesConfig, AGChartAxisConfig
from public.machine_learning.hyperparameters_helper import sweep_hyperparameters
import pandas as pd
from pathlib import Path


@node(parameters=Parameters(force_rerun=True))
async def test_chart_generation(sample_data:PandasModel, **kwargs):
    """
    Simstack node to test chart generation with data from sample_data.csv.
    """
    node_runner = kwargs.get("node_runner")
    # csv_path = Path("private/sample_data.csv")
    # if not csv_path.exists():
    #     return node_runner.fail(f"CSV file not found: {csv_path}")

    df = sample_data.table
    # The CSV has columns: "tree_depth","n_estimators","R2_average"
    
    depths = sorted(df["tree_depth"].unique())
    estimators = sorted(df["n_estimators"].unique())

    # Create Plots
    # Plot 1: R2 vs tree_depth (for each n_estimators)
    depth_all_data = []
    depth_series = []
    for est in estimators:
        y_key = f"r2_est_{est}"
        for _, row in df[df["n_estimators"] == est].iterrows():
            depth_all_data.append({
                "depth": int(row["tree_depth"]),
                y_key: float(row["R2_average"])
            })
        depth_series.append(AGScatterSeriesConfig(xKey="depth", yKey=y_key, title=f"estimators={est}"))

    depth_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Tree Depth (Sample)"),
        data=depth_all_data,
        series=depth_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Tree Depth"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )

    # Plot 2: R2 vs n_estimators (for each tree_depth)
    est_all_data = []
    est_series = []
    for depth in depths:
        y_key = f"r2_depth_{depth}"
        for _, row in df[df["tree_depth"] == depth].iterrows():
            est_all_data.append({
                "estimators": int(row["n_estimators"]),
                y_key: float(row["R2_average"])
            })
        est_series.append(AGScatterSeriesConfig(xKey="estimators", yKey=y_key, title=f"depth={depth}"))

    est_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Number of Estimators (Sample)"),
        data=est_all_data,
        series=est_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Number of Estimators"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )

    node_runner.depth_chart = depth_chart
    node_runner.estimators_chart = est_chart
    
    return node_runner.succeed()


@node(parameters=Parameters(force_rerun=True))
async def optimize_hyperparameters(
    dataset: PandasModel,
    depth_min: IntData,
    depth_max: IntData,
    estimators_min: IntData,
    estimators_max: IntData,
    no_seeds: IntData,
    **kwargs
):
    """
    Simstack node that runs a hyperparameter sweep for the impurity regressor model.
    Calls sweep_hyperparameters and returns the results as attributes of node_runner.
    """
    node_runner = kwargs.get("node_runner")
    
    table, depth_chart, est_chart = await sweep_hyperparameters(
        dataset=dataset,
        depth_min=depth_min,
        depth_max=depth_max,
        estimators_min=estimators_min,
        estimators_max=estimators_max,
        no_seeds=no_seeds,
        **kwargs
    )
    
    # Return data as attributes of node_runner
    node_runner.table = table
    node_runner.depth_chart = depth_chart
    node_runner.estimators_chart = est_chart
    
    return node_runner.succeed()
