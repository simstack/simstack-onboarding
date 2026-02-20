import pandas as pd

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing.data[:30]
y = housing.target[:30]

df = pd.DataFrame(X, columns=housing.feature_names)
df['target'] = y

print(df.shape)
print(df.head())
