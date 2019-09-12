import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.datasets import load_iris


dataIris=load_iris()
dir(dataIris)

df=pd.DataFrame(
    dataIris['data'],
    columns=['SL','SW','PL','PW']
    )

df['target']=dataIris['target']
df['species']=df['target'].apply(lambda x:dataIris['target_names'][x])
# print(df.head())



from sklearn.model_selection import train_test_split
xtrain,xtes,ytrain,ytes=train_test_split(
    dataIris['data'],
    df['target'],
    test_size=.05
    )

# print((xtrain))
# print((ytes))

from sklearn.linear_model import LogisticRegression

model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000)
model.fit(xtrain,ytrain)
prediksi=model.predict(xtes)
# print(model.coef_)
# print(model.intercept_)
# print(prediksi)

dfSetosa=df[df['target']==0]
dfVersicolor=df[df['target']==1]
dfVirginica=df[df['target']==2]


fig=plt.figure('Data Iris',figsize=(16,8))

df['prediksi']=model.predict(df[['SL','SW','PL','PW']])
print(df)

dfPredict1=df[df['prediksi']==0]
dfPredict2=df[df['prediksi']==1]
dfPredict3=df[df['prediksi']==2]

# # Plot Sepal Length vs Sepal Width
# plt.subplot(221)
# plt.title('Data Asli')
# plt.scatter(dfSetosa['SL'],dfSetosa['SW'],marker='o',color='r')
# plt.scatter(dfVersicolor['SL'],dfVersicolor['SW'],marker='o',color='g')
# plt.scatter(dfVirginica['SL'],dfVirginica['SW'],marker='o',color='b')
# plt.xlabel('SL')
# plt.ylabel('SW')
# plt.legend(['Setosa','Versicolor','Virginica'])
# # plt.show()
# plt.subplot(222)
# plt.title('Data Asli')
# plt.scatter(dfSetosa['PL'],dfSetosa['PW'],marker='o',color='r')
# plt.scatter(dfVersicolor['PL'],dfVersicolor['PW'],marker='o',color='g')
# plt.scatter(dfVirginica['PL'],dfVirginica['PW'],marker='o',color='b')
# plt.xlabel('PL')
# plt.ylabel('PW')
# plt.legend(['Setosa','Versicolor','Virginica'])
# # plt.show()
# plt.subplot(223)
# plt.title('Data Prediksi')
# plt.scatter(dfPredict1['SL'],dfPredict1['SW'],marker='o',color='r')
# plt.scatter(dfPredict2['SL'],dfPredict2['SW'],marker='o',color='g')
# plt.scatter(dfPredict3['SL'],dfPredict3['SW'],marker='o',color='b')
# plt.xlabel('SL')
# plt.ylabel('SW')
# plt.legend(['Setosa','Versicolor','Virginica'])
# # plt.show()
# plt.subplot(224)
# plt.title('Data Prediksi')
# plt.scatter(dfPredict1['PL'],dfPredict1['PW'],marker='o',color='r')
# plt.scatter(dfPredict2['PL'],dfPredict2['PW'],marker='o',color='g')
# plt.scatter(dfPredict3['PL'],dfPredict3['PW'],marker='o',color='b')
# plt.xlabel('PL')
# plt.ylabel('PW')
# plt.legend(['Setosa','Versicolor','Virginica'])
# plt.show()