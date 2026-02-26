from simstack.core.node import node
from simstack.models import FloatData


@node
def add(a: FloatData, b: FloatData) -> FloatData:
    return FloatData(field_name="sum", value=a.value + b.value)

@node
def multiply(a: FloatData, b: FloatData) -> FloatData:
    return FloatData(field_name="product", value=a.value * b.value)



