from simstack.core.context import context
from simstack.core.node import node
from simstack.models import Parameters, IntData
from simstack.models.array_storage import ArrayStorage

from public.dataset_ops.plot_one_curve_helper import plot_one_curve_helper

@node(parameters=Parameters(force_rerun=True))
async def plot_one_curve(curve_number: IntData, dataset: ArrayStorage, **kwargs):
    node_runner = kwargs.get("node_runner")
    stress_strain_curves = dataset.get_array() #  get_dataset(dataset_name)
    node_runner.info(f"Loaded {stress_strain_curves.shape} curves")
    strain_data = stress_strain_curves[0,:]
    stress_data = stress_strain_curves[curve_number.value,:]
    chart = await plot_one_curve_helper(stress_data, strain_data, curve_number, **kwargs)
    return chart

