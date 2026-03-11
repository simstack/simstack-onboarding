from typing import Dict, Any

from odmantic import Model
from simstack.models import simstack_model


@simstack_model
class RegressionResult(Model):
    field_name: str
    metrics: Dict[str, Any]

