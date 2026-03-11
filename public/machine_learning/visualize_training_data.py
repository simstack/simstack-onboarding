import asyncio

from odmantic import ObjectId
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters
from simstack.models.charts_artifact import (
    ChartArtifactModel,
    AGBarSeriesConfig,
    AGChartAxisConfig,
    AGChartTitleConfig
)
from simstack.models.pandas_model import PandasModel

from public.machine_learning.make_training_data import make_training_data


@node(parameters=Parameters(force_rerun=True))
async def visualize_impurity_maxima(**kwargs):
    node_runner = kwargs.get("node_runner")
    
    # We can use the training_data if it exists, or the raw summary
    dataset = await context.db.engine.find_one(PandasModel, {"field_name": "training_data"})
    if not dataset:
        node_runner.info("training_data not found, trying raw summary")
        dataset = await context.db.engine.find_one(PandasModel, {"field_name": "synthetic_steel_curve_summary-20260308152912"})
    
    if not dataset:
        raise ValueError("No suitable dataset found for visualization")
    
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
    
    chart.parent_id = ObjectId(kwargs["task_id"])
    await context.db.save(chart)
    node_runner.info(f"Saved impurity maxima bar chart")
    
    node_runner.chart = chart
    return node_runner.succeed()


async def main():
    await context.initialize()
    # await analyze_curve(IntData(field_name="curve_number", value=12))
    await make_training_data()
    await visualize_impurity_maxima()

if __name__ == "__main__":
    asyncio.run(main())