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
df = pd.get_dummies(df, columns = ['Suburb', 'CouncilArea', 'Type'])
x = df.drop('Price',axis=1)
y = df['Price']

# Split Data
X_train, X_test, y_train, y_test = test_train_split(X, y, test_size=0.3, shuffle=True)

# Algorithm and hyperparameters
model = ensemble.GradientBoostingRegressor(
    n_estimators = 150,
    learning_rate = 0.1,
    max_depth = 30,
    min_samples_split = 4,
    min_samples_leaf = 6,
    max_features = 0.6,
    loss = 'huber'
)

model.fit(X_train, y_train)
