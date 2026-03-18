import pprint

from simstack.core.node import node
from simstack.models import IntData, Parameters
from simstack.models.pandas_model import PandasModel
from simstack.models.charts_artifact import ChartArtifactModel, AGChartTitleConfig, AGScatterSeriesConfig, AGChartAxisConfig
from public.machine_learning.hyperparameters_helper import sweep_hyperparameters
import pandas as pd
from pathlib import Path

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
