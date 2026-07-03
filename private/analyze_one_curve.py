#commented out due to being moved to public and the old version of the node and model table creation not ignoring the "private" folder
# from simstack.core.node import node
# from simstack.models import Parameters, IntData, StringData
# from simstack.models.array_storage import ArrayStorage

# from public.dataset_ops.extract_stress_strain_features import extract_stress_strain_features
# from public.dataset_ops.plot_one_curve_helper import plot_one_curve_helper


# @node(parameters=Parameters(force_rerun=True))
# async def analyze_curve(curve_number: IntData, dataset: ArrayStorage, **kwargs):
#     node_runner = kwargs.get("node_runner")
#     stress_strain_curves = dataset.get_array()
#     node_runner.info(f"Loaded {stress_strain_curves.shape} curves")
#     strain_data = stress_strain_curves[0, :]
#     stress_data = stress_strain_curves[curve_number.value, :]

#     node_runner.info(f"Selected curve number: {curve_number.value}")
#     chart = await plot_one_curve_helper(stress_data, strain_data, curve_number, **kwargs)

#     #
#     # From here on is the additional code
#     #
#     strain_stress_model = await extract_stress_strain_features(stress_data, strain_data,curve_number.value)
#     node_runner.info(f"Saved StrainStressModel for curve {curve_number.value}")

#     node_runner.chart = chart
#     node_runner.result = strain_stress_model
#     node_runner.info(f"Analysis result model: {strain_stress_model}")
#     return node_runner.succeed()
