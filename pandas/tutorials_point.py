# Author : Andrew
# Padas is derived from the word : Panel Data

import pandas as pd
import numpy as np

data = np.array(['a', 'b', 'c', 'd', 'e'])
data_dict = {1: 0, 2: 23, '22': 12}

s = pd.Series(data, index=range(100, 100 + len(data)))
s = pd.Series(data_dict)
s = pd.Series(5, index=range(5))  # Create a Series from Scalar

s = pd.DataFrame(data)
data_frame = [['ady', 12],
              ['ada', 11],
              ['asd', 45]]
df = pd.DataFrame(data_frame, columns=['Name', 'Age'])
data = {'Name': ['Tom', 'Jack', 'Steve', 'Ricky'], 'Age': [28, 34, 29, 42]}
df = pd.DataFrame(data)
df = pd.DataFrame(data_frame, columns=['Name', 'Age'], index=range(3))
# Adding a new column to an existing DataFrame object with column label by passing new series
df['Gender'] = [0 for i in range(3)]
df['Sum'] = df['Age'] + df['Gender']
del df['Sum']  # or df.pop('Sum')
# del a row
df.drop(0)

print(df)

# Panel




