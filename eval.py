import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import read_target

df = pd.read_csv('./result.csv', index_col=0)
date = pd.read_csv('./data/date.csv').date.values[106:184]
df = pd.concat([df, pd.DataFrame({'date': date})], axis=1)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
print(df['2018-9'].mean())
