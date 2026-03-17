


import asyncio
from typing import List, Dict, Any
import numpy as np
from simstack.models import IntData, BooleanData
from simstack.models.simple_table import SimpleTable
from simstack.models.pandas_model import PandasModel
from simstack.models.charts_artifact import ChartArtifactModel, AGChartTitleConfig, AGScatterSeriesConfig, AGChartAxisConfig
from models.element_selector import ElementSelector
from private.train_impurity_model import train_impurity_regressor_full


async def sweep_hyperparameters(
    dataset: PandasModel,
    depth_min: IntData,
    depth_max: IntData,
    estimators_min: IntData,
    estimators_max: IntData,
    no_seeds: IntData,
    **kwargs
):
    """
    Runs train_impurity_regressor_full with scaling for a range of tree_depth and number_of_estimators.
    Balances computational effort for 5 different seeds each.
    Collects values in a SimpleTable and plots R2 vs tree-depth and number_of_estimators.
    """
    node_runner = kwargs.get("node_runner")
    
    depth_range = list(range(depth_min.value, depth_max.value + 1, 2))
    estimators_range = list(range(estimators_min.value, estimators_max.value + 1, 20))

    element_selector = ElementSelector(use_C=True, use_Mn=True, use_P=True, use_S=True)
    use_scaling = BooleanData(value=True)
    use_engineered_features = BooleanData(value=False)

    seeds = np.random.randint(0, 10000, size=no_seeds.value)
    
    semaphore = asyncio.Semaphore(10)
    
    async def run_single_train(depth, n_estimators, seed):
        async with semaphore:
            node_runner.info(f"Sweep: depth={depth}, n_estimators={n_estimators}, seed={seed}")
            node_result = await train_impurity_regressor_full(
                dataset=dataset,
                n_estimators=IntData(value=n_estimators),
                max_depth=IntData(value=depth),
                r_seed=IntData(value=seed),
                element_selector=element_selector,
                use_scaling=use_scaling,
                use_engineered_features=use_engineered_features,
                **kwargs
            )
            reg_result = node_result.result
            if reg_result is None:
                raise ValueError(f"Could not find result for depth={depth}, n_estimators={n_estimators}, seed={seed}")
            
            return {
                "depth": depth,
                "n_estimators": n_estimators,
                "seed": seed,
                "r2": reg_result.metrics["average_r2"]
            }

    tasks = []
    for depth in depth_range:
        for n_estimators in estimators_range:
            for seed in seeds:
                tasks.append(run_single_train(depth, n_estimators, seed))

    all_raw_results = await asyncio.gather(*tasks)

    # Aggregate by depth and n_estimators to compute average R2 over seeds
    results_dict = {}
    for res in all_raw_results:
        key = (res["depth"], res["n_estimators"])
        if key not in results_dict:
            results_dict[key] = []
        results_dict[key].append(res["r2"])

    results_list = []
    for (depth, n_estimators), r2_list in results_dict.items():
        avg_r2 = sum(r2_list) / len(r2_list)
        results_list.append({
            "tree_depth": depth,
            "n_estimators": n_estimators,
            "R2_average": avg_r2
        })

    # Create SimpleTable
    table = SimpleTable(name="Hyperparameter Sweep Results")
    table.add_column("tree_depth", "number")
    table.add_column("n_estimators", "number")
    table.add_column("R2_average", "number")
    
    for res in results_list:
        table.add_row(res)
    
    # Create Plots
    # Plot 1: R2 vs tree_depth (for each n_estimators)
    depth_data_map = {}
    for res in results_list:
        d = res["tree_depth"]
        if d not in depth_data_map:
            depth_data_map[d] = {"depth": d}
        depth_data_map[d][f"r2_est_{res['n_estimators']}"] = res["R2_average"]
    
    depth_all_data = sorted(depth_data_map.values(), key=lambda x: x["depth"])
    depth_series = [
        AGScatterSeriesConfig(xKey="depth", yKey=f"r2_est_{est}", title=f"estimators={est}")
        for est in estimators_range
    ]

    depth_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Tree Depth"),
        data=depth_all_data,
        series=depth_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Tree Depth"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )

    # Plot 2: R2 vs n_estimators (for each tree_depth)
    est_data_map = {}
    for res in results_list:
        e = res["n_estimators"]
        if e not in est_data_map:
            est_data_map[e] = {"estimators": e}
        est_data_map[e][f"r2_depth_{res['tree_depth']}"] = res["R2_average"]

    est_all_data = sorted(est_data_map.values(), key=lambda x: x["estimators"])
    est_series = [
        AGScatterSeriesConfig(xKey="estimators", yKey=f"r2_depth_{depth}", title=f"depth={depth}")
        for depth in depth_range
    ]

    est_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="R2 vs Number of Estimators"),
        data=est_all_data,
        series=est_series,
        axes=[
            AGChartAxisConfig(type="number", position="bottom", title="Number of Estimators"),
            AGChartAxisConfig(type="number", position="left", title="Average R2"),
        ]
    )


    
    return table, depth_chart, est_chart
