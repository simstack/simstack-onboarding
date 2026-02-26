import pandas as pd
import cloudpickle
from typing import Optional
from odmantic import Model
from simstack.models.pickle_models import _BytesB64Mixin


class DataFrameModel(_BytesB64Mixin, Model):
    field_name: str
    data_bytes: Optional[bytes] = None

    def store_df(self, df: pd.DataFrame):
        self.data_bytes = cloudpickle.dumps(df)

    def load_df(self) -> pd.DataFrame:
        if self.data_bytes is None:
            raise ValueError("No data stored")
        return cloudpickle.loads(self.data_bytes)
