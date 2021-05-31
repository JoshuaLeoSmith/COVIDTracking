import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('drive/My Drive/Colab Notebooks/LinearRegression/trainingdata3.csv')
dataset



ind = pd.DataFrame(preprocessing.scale(dataset.iloc[:,:-1]))
dep = pd.DataFrame(dataset.iloc[:,-1])

x = pd.DataFrame(preprocessing.scale(dataset.iloc[:,:-1])).values.reshape(-1,1)
y = pd.DataFrame(dataset.iloc[:,-1]).values.reshape(-1,1)

ind_train, ind_test, dep_train, dep_test = train_test_split(ind, dep, test_size=0.2, random_state=1)


regressor = LinearRegression()
regressor.fit(ind_train, dep_train)

v = pd.DataFrame(regressor.coef_,index=['Co-efficient']).transpose()
w = pd.DataFrame(ind.columns, columns=['Attribute'])

coeff_df = pd.concat([w,v], axis=1, join='inner')

dep_pred = regressor.predict(ind_test)
dep_pred = pd.DataFrame(dep_pred, columns=['Predicted'])
