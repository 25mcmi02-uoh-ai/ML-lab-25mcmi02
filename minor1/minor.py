import pandas as pd
import numpy as np

df = pd.read_csv("seattle-weather.csv")

print("Sample of the Data:")
print(df.head())

print(f"size of data: {df.shape}")
print("The following is distribution of the target:")
print(df["  "])