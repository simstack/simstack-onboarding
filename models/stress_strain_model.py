from typing import Dict, Any, Optional

from odmantic import Model
from simstack.models import simstack_model


@simstack_model
class StrainStressModel(Model):
    curve_index: int
    linear_region: Dict[str, Any]
    yield_strength: Optional[Dict[str, Any]]
    ultimate_strength: Dict[str, Any]
    fracture: Dict[str, Any]
