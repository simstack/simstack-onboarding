from odmantic import ObjectId
from simstack.core.context import context
from simstack.core.node_runner import NodeRunner
from simstack.models import IntData
from simstack.models.array_storage import ArrayStorage
from simstack.models.charts_artifact import create_simple_line_chart


async def plot_one_curve_helper(stress_data: ArrayStorage, strain_data: ArrayStorage, curve_number: IntData, **kwargs):
    node_runner = kwargs.get("node_runner")
    data = []
    for stress, strain in zip(stress_data.tolist(), strain_data.tolist()):
        data.append({"Stress": stress, "Strain": strain})

    chart = create_simple_line_chart(
        data=data,
        x_key="Strain",
        y_key="Stress",
        title=f"Stress vs Strain - Curve {curve_number.value}"
    )
    chart.parent_id = ObjectId(kwargs["task_id"])
    await context.db.save(chart)
    return chart
