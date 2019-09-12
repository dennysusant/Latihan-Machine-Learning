import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris

data=load_iris()
df=pd.DataFrame(
    data['data'],
    columns=['SL','SW','PL','PW']
)
df['target']=data['target']
df['spesies']=df['target'].apply(
    lambda x:data['target_names'][x]
)
# print(df.head())
#Splitting =5%test
from sklearn.model_selection import train_test_split
xtrain,xtes,ytrain,ytes=train_test_split(
    df[['SL','SW','PL','PW']],
    df['target'],
    test_size=.05
    )

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs')
model.fit(xtrain,ytrain)
print(model.score(xtrain,ytrain))
print(xtes)
print(ytes)
print(model.predict(xtes))
# print(model.predict_proba(xtes))