import asyncio
import pandas as pd

from typing import Optional, List, Dict, Any
from odmantic import ObjectId, Model, EmbeddedModel
from simstack.core import node_runner
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import IntData, Parameters, simstack_model
from simstack.models.array_storage import ArrayStorage
from simstack.models.charts_artifact import (
    ChartArtifactModel,
    create_simple_line_chart,
    AGBarSeriesConfig,
    AGChartAxisConfig,
    AGChartTitleConfig
)
from simstack.models.pandas_model import PandasModel

from private.np_help import extract_stress_strain_features


@node(parameters=Parameters(force_rerun=True))
async def read_np_dataset(curve_number: IntData, **kwargs):
    node_runner = kwargs.get("node_runner")
    np_array_input = await context.db.engine.find_one(ArrayStorage, {"name": "synthetic_steel_stress_strain_curves-20260308141111"})
    if np_array_input is None:
        raise ValueError("ArrayStorage not found")
    stress_strain_curves = np_array_input.get_array()
    node_runner.info(f"Loaded {stress_strain_curves.shape} curves")
    strain_data = stress_strain_curves[0,:]
    stress_data = stress_strain_curves[curve_number.value,:]
    
    node_runner.info(f"Selected curve number: {curve_number.value}")

    data = []
    for stress, strain in zip(stress_data.tolist(), strain_data.tolist()):
        data.append({"Stress": stress, "Strain": strain})

    node_runner.info(f"Created stress-strain chart artifact {data[:20]}")
    chart = create_simple_line_chart(
        data=data,
        x_key="Strain",
        y_key="Stress",
        title=f"Stress vs Strain - Curve {curve_number.value}"
    )
    chart.parent_id = ObjectId(kwargs["task_id"])
    await context.db.save(chart)
    node_runner.info(f"Saved stress-strain chart artifact")

    return chart

@simstack_model
class StrainStressModel(Model):
    curve_index: int
    linear_region: Dict[str, Any]
    yield_strength: Optional[Dict[str, Any]]
    ultimate_strength: Dict[str, Any]
    fracture: Dict[str, Any]


@node(parameters=Parameters(force_rerun=True))
async def analyze_curve(curve_number: IntData, **kwargs):
    node_runner = kwargs.get("node_runner")
    np_array_input = await context.db.engine.find_one(ArrayStorage,
                                                      {"name": "synthetic_steel_stress_strain_curves-20260308141111"})
    if np_array_input is None:
        raise ValueError("ArrayStorage not found")
    stress_strain_curves = np_array_input.get_array()
    node_runner.info(f"Loaded {stress_strain_curves.shape} curves")
    strain_data = stress_strain_curves[0, :]
    stress_data = stress_strain_curves[curve_number.value, :]

    node_runner.info(f"Selected curve number: {curve_number.value}")

    data = []
    for stress, strain in zip(stress_data.tolist(), strain_data.tolist()):
        data.append({"Stress": stress, "Strain": strain})

    node_runner.info(f"Created stress-strain chart artifact {data[:20]}")
    chart = create_simple_line_chart(
        data=data,
        x_key="Strain",
        y_key="Stress",
        title=f"Stress vs Strain - Curve {curve_number.value}"
    )
    chart.parent_id = ObjectId(kwargs["task_id"])
    await context.db.save(chart)
    node_runner.info(f"Saved stress-strain chart artifact")

    analysis_result = extract_stress_strain_features(stress_data, strain_data)
    node_runner.info(f"Analysis result: {analysis_result}")

    # Pack analysis results into StrainStressModel
    strain_stress_model = StrainStressModel(
        curve_index=curve_number.value,
        linear_region=analysis_result["linear_region"],
        yield_strength=analysis_result.get("yield_strength"),
        ultimate_strength=analysis_result["ultimate_strength"],
        fracture=analysis_result["fracture"]
    )
    await context.db.save(strain_stress_model)
    node_runner.info(f"Saved StrainStressModel for curve {curve_number.value}")

    node_runner.chart = chart
    node_runner.result = strain_stress_model
    node_runner.info(f"Analysis result model: {strain_stress_model}")
    return node_runner.succeed()



@node(parameters=Parameters(force_rerun=True))
async def make_training_data(**kwargs):
    node_runner = kwargs.get("node_runner")
    np_array_input = await context.db.engine.find_one(ArrayStorage,
                                                      {"name": "synthetic_steel_stress_strain_curves-20260308141111"})
    if np_array_input is None:
        raise ValueError("ArrayStorage not found")
    stress_strain_curves = np_array_input.get_array()

    parameter_dataset = await context.db.engine.find_one(PandasModel,
                                                      {"field_name": "synthetic_steel_curve_summary-20260308152912"})

    if not parameter_dataset:
        raise ValueError("PandasModel not found")

    df = parameter_dataset.table
    strain_data = stress_strain_curves[0, :]

    results = []

    for index, row in df.iterrows():
        curve_index = index + 1 # First curve is at index 1 in stress_strain_curves
        if curve_index >= stress_strain_curves.shape[0]:
            break
            
        stress_data = stress_strain_curves[curve_index, :]

        node_runner.info(f"Analyzing curve number: {curve_index}")
        analysis_result = extract_stress_strain_features(stress_data, strain_data)

        # Combine impurity data with extracted features
        record = {
            "C_wt_percent": row["C_wt_percent"],
            "Mn_wt_percent": row["Mn_wt_percent"],
            "P_wt_percent": row["P_wt_percent"],
            "S_wt_percent": row["S_wt_percent"],
            "youngs_modulus_MPa": analysis_result["linear_region"]["youngs_modulus_MPa"],
            "yield_strength_MPa": analysis_result["yield_strength"].get("stress_MPa"),
            "ultimate_strength_MPa": analysis_result["ultimate_strength"].get("stress_MPa"),
            "fracture_stress_MPa": analysis_result["fracture"]["stress_MPa"],
            "fracture_strain": analysis_result["fracture"]["strain"]
        }
        results.append(record)

    training_df = pd.DataFrame(results)
    
    training_data_model = PandasModel(field_name="training_data")
    training_data_model.table = training_df
    
    await context.db.save(training_data_model)
    node_runner.info(f"Saved training_data dataset with {len(training_df)} rows")

    node_runner.result = training_data_model
    return node_runner.succeed()


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
    # await make_training_data()
    await visualize_impurity_maxima()

if __name__ == "__main__":
    asyncio.run(main())