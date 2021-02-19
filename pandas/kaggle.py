import pandas as pd
import sidetable


wine_reviews = pd.read_csv('winemag-data-130k-v2.csv')
df = pd.DataFrame(wine_reviews, copy=True)
df.country  # == df['country']  == df.iloc[:,1]

region = pd.DataFrame(df.country + ' - ' + df.region_1, copy=True, columns=['region'])
region.rename(columns={'region': 'Region'}, inplace=True)
region.rename(columns=lambda x: 'Region_', inplace=True)

