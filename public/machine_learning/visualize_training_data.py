import asyncio

from odmantic import ObjectId
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters
from simstack.models.charts_artifact import (
    ChartArtifactModel,
    AGBarSeriesConfig,
    AGHeatmapSeriesConfig,
    AGRangeBarSeriesConfig,
    AGChartAxisConfig,
    AGChartTitleConfig,
    create_simple_heatmap_chart,
)
from simstack.models.table_artifact import TableArtifactModel, AGGridColumnDef
from simstack.models.pandas_model import PandasModel

from simstack.models.charts_artifact import create_simple_scatter_chart


async def _save_correlation_heatmap(
    corr_frame,
    title: str,
    task_id,
    node_runner,
    charts: list,
    x_name: str = "Column",
    y_name: str = "Row",
):
    heatmap_data = []
    for row_name in corr_frame.index:
        for col_name in corr_frame.columns:
            heatmap_data.append(
                {
                    "x_feature": str(col_name),
                    "y_feature": str(row_name),
                    "correlation": float(corr_frame.loc[row_name, col_name]),
                }
            )

    heatmap_chart = create_simple_heatmap_chart(
        data=heatmap_data,
        x_key="x_feature",
        y_key="y_feature",
        color_key="correlation",
        title=title,
        parent_id=ObjectId(task_id) if task_id else None,
    )

    if heatmap_chart.series and isinstance(heatmap_chart.series[0], AGHeatmapSeriesConfig):
        heatmap_chart.series[0].xName = x_name
        heatmap_chart.series[0].yName = y_name
        heatmap_chart.series[0].colorName = "Pearson correlation"
        heatmap_chart.series[0].colorDomain = [-1.0, 1.0]
        heatmap_chart.series[0].colorRange = ["#2166ac", "#f7f7f7", "#b2182b"]

    await context.db.save(heatmap_chart)
    charts.append(heatmap_chart)
    node_runner.info(f"Saved heatmap: {title}")


async def _visualize_strain_vs_concentration_internal(dataset: PandasModel, **kwargs):
    node_runner = kwargs.get("node_runner")
    task_id = kwargs.get("task_id")
    
    df = dataset.table
    impurity_cols = ["C_wt_percent", "Mn_wt_percent", "P_wt_percent", "S_wt_percent"]
    
    # We want to analyze strains: fracture_strain and uniform_strain
    strain_cols = ["fracture_strain"]
    if "uniform_strain" in df.columns:
        strain_cols.append("uniform_strain")
    
    node_runner.info(f"Visualizing relationship between {impurity_cols} and {strain_cols}")

    charts = []
    for strain in strain_cols:
        for impurity in impurity_cols:
            plot_data = []
            for _, row in df.iterrows():
                plot_data.append({
                    impurity: float(row[impurity]),
                    strain: float(row[strain])
                })
            
            title = f"{strain} vs {impurity}"
            chart = create_simple_scatter_chart(
                data=plot_data,
                x_key=impurity,
                y_key=strain,
                title=title,
                parent_id=ObjectId(task_id) if task_id else None
            )
            await context.db.save(chart)
            charts.append(chart)
            node_runner.info(f"Saved scatter chart: {title}")

    # Create correlation matrix as a table
    corr = df[impurity_cols + strain_cols].corr()
    node_runner.info("Correlation Matrix:\n" + str(corr))

    # Create TableArtifactModel for correlation matrix
    column_defs = [AGGridColumnDef(field="feature", headerName="Feature")]
    for col in corr.columns:
        column_defs.append(AGGridColumnDef(field=col, headerName=col))

    row_data = []
    for row_name in corr.index:
        row_dict = {"feature": row_name}
        for col in corr.columns:
            row_dict[col] = float(corr.loc[row_name, col])
        row_data.append(row_dict)

    # Use a dummy ObjectId if task_id is None to avoid pydantic validation error in TableArtifactModel
    # parent_id in TableArtifactModel is not Optional in this version of simstack
    effective_parent_id = ObjectId(task_id) if task_id else ObjectId()

    corr_table = TableArtifactModel(
        columns_defs=column_defs,
        row_data=row_data,
        parent_id=effective_parent_id
    )

    await context.db.save(corr_table)
    charts.append(corr_table)
    node_runner.info("Saved correlation matrix table")

    await _save_correlation_heatmap(
        corr,
        title="Impurity / Strain Correlation Heatmap",
        task_id=task_id,
        node_runner=node_runner,
        charts=charts,
        x_name="Variable (X)",
        y_name="Variable (Y)",
    )

    model_feature_cols = [
        col
        for col in [
            "youngs_modulus_MPa",
            "yield_strength_MPa",
            "ultimate_strength_MPa",
            "fracture_stress_MPa",
            "fracture_strain",
            "uniform_strain",
        ]
        if col in df.columns
    ]

    if len(model_feature_cols) >= 2:
        feature_corr = df[model_feature_cols].corr()
        await _save_correlation_heatmap(
            feature_corr,
            title="ML Input Feature Correlation Heatmap",
            task_id=task_id,
            node_runner=node_runner,
            charts=charts,
            x_name="Model input feature (X)",
            y_name="Model input feature (Y)",
        )

    if model_feature_cols:
        feature_target_corr = df[model_feature_cols + impurity_cols].corr().loc[model_feature_cols, impurity_cols]
        await _save_correlation_heatmap(
            feature_target_corr,
            title="ML Feature-to-Target Correlation Heatmap",
            task_id=task_id,
            node_runner=node_runner,
            charts=charts,
            x_name="Impurity target",
            y_name="Model input feature",
        )

    engineered_feature_data = {}
    if all(col in df.columns for col in ("ultimate_strength_MPa", "yield_strength_MPa")):
        engineered_feature_data["strength_ratio"] = df["ultimate_strength_MPa"] / df["yield_strength_MPa"]
    if all(col in df.columns for col in ("fracture_strain", "ultimate_strength_MPa", "youngs_modulus_MPa")):
        engineered_feature_data["strain_margin"] = (
            df["fracture_strain"] - (df["ultimate_strength_MPa"] / df["youngs_modulus_MPa"])
        )

    if engineered_feature_data:
        engineered_df = df[impurity_cols].copy()
        for col_name, series in engineered_feature_data.items():
            engineered_df[col_name] = series
        engineered_cols = list(engineered_feature_data.keys())
        engineered_corr = engineered_df[engineered_cols + impurity_cols].corr().loc[engineered_cols, impurity_cols]
        await _save_correlation_heatmap(
            engineered_corr,
            title="Engineered Feature-to-Target Correlation Heatmap",
            task_id=task_id,
            node_runner=node_runner,
            charts=charts,
            x_name="Impurity target",
            y_name="Engineered feature",
        )

    if hasattr(node_runner, 'result'):
        node_runner.result = {"charts_count": len(charts), "correlation_matrix": corr.to_dict()}
    return charts


@node(parameters=Parameters(force_rerun=True))
async def visualize_strain_vs_concentration(dataset: PandasModel, **kwargs):
    return await _visualize_strain_vs_concentration_internal(dataset, **kwargs)


async def _visualize_impurity_ranges_internal(dataset: PandasModel, **kwargs):
    node_runner = kwargs.get("node_runner")
    task_id = kwargs.get("task_id")

    df = dataset.table
    impurity_cols = ["C_wt_percent", "Mn_wt_percent", "P_wt_percent", "S_wt_percent"]

    min_values = df[impurity_cols].min()
    max_values = df[impurity_cols].max()

    chart_data = []
    for col in impurity_cols:
        chart_data.append({
            "impurity": col.split("_")[0],
            "min_value": float(min_values[col]),
            "max_value": float(max_values[col])
        })

    range_series = [
        AGRangeBarSeriesConfig(
            type="range-bar",
            xKey="impurity",
            yLowKey="min_value",
            yHighKey="max_value",
            title="Impurity Concentration Range",
            data=chart_data
        )
    ]

    axes = [
        AGChartAxisConfig(type="category", position="bottom", title="Impurity"),
        AGChartAxisConfig(type="number", position="left", title="Concentration (wt%)")
    ]

    range_chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="Impurity Concentration Min/Max Ranges"),
        series=range_series,
        axes=axes,
        data=chart_data
    )

    if task_id:
        range_chart.parent_id = ObjectId(task_id)
    await context.db.save(range_chart)
    node_runner.info("Saved impurity min/max range chart")

    return range_chart


async def _visualize_impurity_maxima_internal(dataset: PandasModel, **kwargs):
    node_runner = kwargs.get("node_runner")
    task_id = kwargs.get("task_id")
    
    # # We can use the training_data if it exists, or the raw summary
    # dataset = await context.db.engine.find_one(PandasModel, {"field_name": "training_data"})
    # if not dataset:
    #     node_runner.info("training_data not found, trying raw summary")
    #     dataset = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": "synthetic_steel_curve_summary"}})
    #
    # if not dataset:
    #     raise ValueError("No suitable dataset found for visualization")
    
    df = dataset.table
    impurity_cols = ["C_wt_percent", "Mn_wt_percent", "P_wt_percent", "S_wt_percent"]
    
    # Calculate maxima
    max_values = df[impurity_cols].max()
    
    # Format data for chart: list of { "impurity": "C", "max_value": 0.3 }
    chart_data = []
    for col in impurity_cols:
        chart_data.append({
            "impurity": col.split("_")[0], # Just the element name
            "max_value": float(max_values[col])
        })
    
    node_runner.info(f"Chart data: {chart_data}")
    
    # Create ChartArtifactModel with AGBarSeriesConfig
    series = [
        AGBarSeriesConfig(
            type="bar",
            xKey="max_value",
            yKey="impurity",
            title="Maximum Impurity Concentration",
            data=chart_data
        )
    ]
    
    axes = [
        AGChartAxisConfig(type="category", position="left", title="Impurity"),
        AGChartAxisConfig(type="number", position="bottom", title="Max Concentration (wt%)")
    ]
    
    chart = ChartArtifactModel(
        title=AGChartTitleConfig(text="Maximum Impurity Concentrations"),
        series=series,
        axes=axes,
        data=chart_data
    )
    
    if task_id:
        chart.parent_id = ObjectId(task_id)
    await context.db.save(chart)
    node_runner.info(f"Saved impurity maxima bar chart")
    
    return chart


@node(parameters=Parameters(force_rerun=True))
async def visualize_impurity_maxima(dataset: PandasModel, **kwargs):
    max_chart = await _visualize_impurity_maxima_internal(dataset, **kwargs)

    # We can remove old max_chart and set this new range_chart as node_runner.chart if needed
    range_chart = await _visualize_impurity_ranges_internal(dataset, **kwargs)

    node_runner = kwargs.get("node_runner")
    node_runner.chart = max_chart
    return node_runner.succeed()


async def main():
    await context.initialize()
    
    # Try to find the training data
    dataset = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": "training_data"}})
    if not dataset:
        print("INFO: training_data not found, trying raw summary")
        dataset = await context.db.engine.find_one(PandasModel, {"field_name": {"$regex": "synthetic_steel_curve_summary"}})
    
    if not dataset:
        raise ValueError("No suitable dataset found for visualization")

    # Mock NodeRunner for manual execution
    class MockNodeRunner:
        def __init__(self):
            self.result = None
            self.chart = None
        def info(self, msg): print(f"INFO: {msg}")
        def succeed(self): print("SUCCESS")
        def fail(self, msg): print(f"FAIL: {msg}")

    runner = MockNodeRunner()
    kwargs = {"node_runner": runner, "task_id": None}
    
    print("\n--- Visualizing Strain vs Concentration ---")
    await _visualize_strain_vs_concentration_internal(dataset, **kwargs)
    
    print("\n--- Visualizing Impurity Maxima ---")
    await _visualize_impurity_maxima_internal(dataset, **kwargs)
    await _visualize_impurity_ranges_internal(dataset, **kwargs)

if __name__ == "__main__":
    asyncio.run(main())
