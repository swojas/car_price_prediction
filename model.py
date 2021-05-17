import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

dataset = pd.read_csv('car data.csv')

dataset['diff'] = 2021 - dataset['Year']
dataset.drop(['Car_Name', 'Year'], axis=1, inplace=True)
dataset = pd.get_dummies(dataset, drop_first=True)

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred) * 100)

pickle.dump(regressor, open('model.pkl', 'wb'))