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
