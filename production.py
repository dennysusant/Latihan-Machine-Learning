import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import seaborn as sb
from sklearn import linear_model
from sklearn.model_selection import train_test_split


datarumah=fetch_california_housing()
df=pd.DataFrame(
    datarumah['data'],
    columns=datarumah['feature_names'])
df['target']=datarumah['target']

x=df[['MedInc',
 'HouseAge',
 'AveRooms',
 'AveBedrms',
 'Population',
 'AveOccup',
 'Latitude',
 'Longitude']]
y=df['target']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1)
x_test.shape

model=linear_model.LinearRegression()
model.fit(x_train,y_train)

# #Save model ML:pickle
# import pickle 
# with open ('modelPickle.pkl','wb') as modelku:
#     pickle.dump(model,modelku)

#============Joblib========Lebih efektif bila data banyak numpy array
import joblib
joblib.dump(model,'modelJoblib.joblib')
