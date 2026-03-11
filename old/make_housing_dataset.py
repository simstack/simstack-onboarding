import asyncio
import pandas as pd
from simstack.core.context import context
from simstack.core.node import node
from simstack.models import IntData
from sklearn.datasets import fetch_california_housing

from public.models.df_model import DataFrameModel

@node
async def make_housing_dataset(number: IntData,**kwargs):
    max_number = number.value
    if max_number > 100:
        max_number = 100
    if max_number < 1:
        max_number = 1
    housing = fetch_california_housing()
    X = housing.data[:max_number]
    y = housing.target[:max_number]

    df = pd.DataFrame(X, columns=housing.feature_names)
    df['target'] = y

    df_model = DataFrameModel(field_name="housing_dataset")
    df_model.store_df(df)
    await context.db.save(df_model)
    return True

if __name__ == "__main__":
    asyncio.run(make_housing_dataset())