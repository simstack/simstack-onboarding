import asyncio
import pandas as pd
from docutils.nodes import field_name
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters, StringData, IntData
from simstack.models.array_storage import ArrayStorage
from simstack.models.pandas_model import PandasModel

from public.dataset_ops.extract_stress_strain_features import extract_stress_strain_features


async def _process_batch_internal(curves_dataset: ArrayStorage, parameter_dataset: PandasModel, batch_start: IntData, batch_end: IntData, **kwargs):
    node_runner = kwargs.get("node_runner")
    node_runner.info(f"Processing batch {batch_start.value} to {batch_end.value}")
    stress_strain_curves = curves_dataset.get_array()

    df = parameter_dataset.table
    strain_data = stress_strain_curves[0, :]

    results = []
    # Process only the slice [batch_start, batch_end)
    for index in range(batch_start.value, min(batch_end.value, len(df))):
        row = df.iloc[index]
        curve_index = index + 1 # First curve is at index 1 in stress_strain_curves
        if curve_index >= stress_strain_curves.shape[0]:
            break

        stress_data = stress_strain_curves[curve_index, :]

        analysis_result = await extract_stress_strain_features(stress_data, strain_data, index)

        # Combine impurity data with extracted features
        record = {
            "C_wt_percent": row["C_wt_percent"],
            "Mn_wt_percent": row["Mn_wt_percent"],
            "P_wt_percent": row["P_wt_percent"],
            "S_wt_percent": row["S_wt_percent"],
            "youngs_modulus_MPa": analysis_result.linear_region["youngs_modulus_MPa"],
            "yield_strength_MPa": analysis_result.yield_strength.get("stress_MPa"),
            "ultimate_strength_MPa": analysis_result.ultimate_strength.get("stress_MPa"),
            "fracture_stress_MPa": analysis_result.fracture["stress_MPa"],
            "fracture_strain": analysis_result.fracture["strain"]
        }
        results.append(record)
    
    # Pack result in a PandasModel to be returned from node
    batch_df = pd.DataFrame(results)
    batch_result_model = PandasModel(field_name=f"{curves_dataset.name}_batch_{batch_start.value}")
    batch_result_model.table = batch_df
    return batch_result_model


@node
async def process_batch(curces_dataset: ArrayStorage, parameters_dataset: PandasModel, batch_start: IntData, batch_end: IntData, **kwargs):
    return await _process_batch_internal(curces_dataset, parameters_dataset, batch_start, batch_end, **kwargs)


@node(parameters=Parameters(force_rerun=True))
async def make_training_data(curves_dataset: ArrayStorage, parameter_dataset: PandasModel, **kwargs):
    node_runner = kwargs.get("node_runner")

    df = parameter_dataset.table
    total_records = len(df)
    batch_size = 100
    concurrency_limit = 10
    
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_batch_with_semaphore(start, end):
        async with semaphore:
            # Call the internal function directly to avoid @node scheduling issues in standalone/test
            return await process_batch(curves_dataset, parameter_dataset, IntData(field_name="batch_start", value=start),
                                       IntData(field_name="batch_end", value=end), **kwargs)

    tasks = []
    for i in range(0, total_records, batch_size):
        tasks.append(run_batch_with_semaphore(i, i + batch_size))

    node_runner.info(f"Starting {len(tasks)} batches for {total_records} records with concurrency {concurrency_limit}")
    
    batch_results = await asyncio.gather(*tasks)
    
    # Combine results
    all_dfs = [model.table for model in batch_results if hasattr(model, 'table')]
    if not all_dfs:
        return node_runner.fail("No results generated")
        
    training_df = pd.concat(all_dfs, ignore_index=True)

    training_data_model = PandasModel(field_name=dataset.field_name + "training_data")
    training_data_model.table = training_df

    await context.db.save(training_data_model)
    node_runner.info(f"Saved training_data dataset with {len(training_df)} rows")

    node_runner.result = training_data_model
    return node_runner.succeed()
