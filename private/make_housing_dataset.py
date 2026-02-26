import pandas as pd

from sklearn.datasets import fetch_california_housing


if __name__ != "__main__":
    housing = fetch_california_housing()
    X = housing.data[:30]
    y = housing.target[:30]

    df = pd.DataFrame(X, columns=housing.feature_names)
    df['target'] = y

    print(df.shape)
    print(df.head())
