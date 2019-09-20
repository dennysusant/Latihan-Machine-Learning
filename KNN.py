import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


dataIris=load_iris()
# print(dir(dataIris))
df=pd.DataFrame(
    dataIris['data'],
    columns=['SL','SW','PL','PW']
)
df['target']=dataIris['target']
df['spesies']=df['target'].apply(
    lambda x : dataIris['target_names'][x]
)


from sklearn.model_selection import train_test_split
xtrain,xtes,ytrain,ytes=train_test_split(
    dataIris['data'],
    df['target'],
    test_size=.05
    )
# print(len(xtrain))
# print(len(xtes))

#KNN
#Mengukur K di KKN:
#1. cari akar => sqrt(total data point) =>(150)(dari total data iris)
#2.=> ganjil (+1 bila sqrt genap)
def nilai_k():
    k=round((len(xtrain)+len(xtes))**.5)
    if k%2==0:
        return k+1
    else:
        return k
# print(nilai_k())
model=KNeighborsClassifier(
    n_neighbors=nilai_k()
)
model.fit(xtrain,ytrain)
# print(model.score(xtrain,ytrain))
# pred=model.predict([xtes[0]])
# print(model.score(xtes,ytes))
# print(pred)
# print(ytes.iloc[0])
datapred=model.predict(dataIris['data'])
df['prediksi']=datapred


plt.subplot(221)
plt.plot(
    df[df['target']==0]['PL'],
    df[df['target']==0]['PW'],
    'yo'
)
plt.plot(
    df[df['target']==1]['PL'],
    df[df['target']==1]['PW'],
    'ro'
)
plt.plot(
    df[df['target']==2]['PL'],
    df[df['target']==2]['PW'],
    'bo'
)
plt.plot(
    df[df['target']==3]['PL'],
    df[df['target']==3]['PW'],
    'go'
)


plt.subplot(222)
plt.plot(
    df[df['prediksi']==0]['PL'],
    df[df['prediksi']==0]['PW'],
    'yo'
)
plt.plot(
    df[df['prediksi']==1]['PL'],
    df[df['prediksi']==1]['PW'],
    'ro'
)
plt.plot(
    df[df['prediksi']==2]['PL'],
    df[df['prediksi']==2]['PW'],
    'bo'
)
plt.plot(
    df[df['prediksi']==3]['PL'],
    df[df['prediksi']==3]['PW'],
    'go'
)

plt.subplot(223)
plt.plot(
    df[df['target']==0]['SL'],
    df[df['target']==0]['SW'],
    'yo'
)
plt.plot(
    df[df['target']==1]['SL'],
    df[df['target']==1]['SW'],
    'ro'
)
plt.plot(
    df[df['target']==2]['SL'],
    df[df['target']==2]['SW'],
    'bo'
)
plt.plot(
    df[df['target']==3]['SL'],
    df[df['target']==3]['SW'],
    'go'
)
plt.subplot(224)
plt.plot(
    df[df['prediksi']==0]['SL'],
    df[df['prediksi']==0]['SW'],
    'yo'
)
plt.plot(
    df[df['prediksi']==1]['SL'],
    df[df['prediksi']==1]['SW'],
    'ro'
)
plt.plot(
    df[df['prediksi']==2]['SL'],
    df[df['prediksi']==2]['SW'],
    'bo'
)
plt.plot(
    df[df['prediksi']==3]['SL'],
    df[df['prediksi']==3]['SW'],
    'go'
)
# plt.plot(
#     df.index,df['target'],'r-',
#     df.index,df['prediksi'],'b-')
plt.show()

