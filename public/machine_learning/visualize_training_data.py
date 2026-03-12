import asyncio

from odmantic import ObjectId
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters
from simstack.models.charts_artifact import (
    ChartArtifactModel,
    AGBarSeriesConfig,
    AGRangeBarSeriesConfig,
    AGChartAxisConfig,
    AGChartTitleConfig
)
from simstack.models.table_artifact import TableArtifactModel, AGGridColumnDef
from simstack.models.pandas_model import PandasModel

from simstack.models.charts_artifact import create_simple_scatter_chart


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

if __name__ == "__main__":
    asyncio.run(main())
