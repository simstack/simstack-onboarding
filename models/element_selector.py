from odmantic import Model, Field
from simstack.models import simstack_model


@simstack_model
class ElementSelector(Model):
    use_Mn: bool  = Field(default=False)
    use_S: bool = Field(default=False)
    use_P: bool = Field(default=False)
    use_C: bool = Field(default=False)