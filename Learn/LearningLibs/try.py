import pandas as pd
import numpy as np

df = pd.DataFrame( { "0": [2, 3], "1": [1, 4], "2": [4, 2] })
df.index = ["x", "y"]

total_sum = np.sum(df.values)

print(total_sum)