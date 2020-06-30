# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensamble
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('data/Melbourne_housing_FULL.csv')
# Scrub Data
del df['Adress']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']

df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
df.
