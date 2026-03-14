from typing import Dict, Any

from odmantic import Model
from simstack.core.node import node
from simstack.models import FloatData, IntData, simstack_model


@node
def add(a: FloatData, b: FloatData, **kwargs) -> FloatData:
    return FloatData(field_name="sum", value=a.value + b.value)

@node
def multiply(a: FloatData, b: FloatData, **kwargs) -> FloatData:
    return FloatData(field_name="product", value=a.value * b.value)

@simstack_model
class JoergModel(Model):
    structure: Dict[str, Any]

@node
def first_node(**kwargs):
    node_runner = kwargs.get("node_runner")
    node_runner.float_result = FloatData(field_name="result", value=1.0)
    node_runner.int_result = IntData(field_name="result", value=1)
    my_joerg = JoergModel(structure={"a": 1, "b": 2})
    node_runner.joerg_result = my_joerg
    return node_runner.succeed()

def second_node(**kwargs):
    result = first_node(**kwargs)
    print(result.float_result.value)
    print(result.joerg_result.structure)

def third_node(**kwargs):
        result = first_node(**kwargs)
        print(result.float_result.value)
        print(result.joerg_result.structure)