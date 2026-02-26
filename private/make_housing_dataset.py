import asyncio

import pandas as pd
from simstack.core.context import context
from sklearn.datasets import fetch_california_housing

from public.models.df_model import DataFrameModel


async def main():
    await context.initialize()

    housing = fetch_california_housing()
    X = housing.data[:30]
    y = housing.target[:30]

    df = pd.DataFrame(X, columns=housing.feature_names)
    df['target'] = y

    df_model = DataFrameModel(field_name="housing_dataset")
    df_model.store_df(df)
    await context.db.save(df_model)

if __name__ == "__main__":
    asyncio.run(main())