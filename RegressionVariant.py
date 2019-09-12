import numpy as np 
import pandas as pd 
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

data=np.random.RandomState(1)
# print(data.rand(10))
# x=5 * data.rand(50)
# y=2 * x + data.randn(50)
x=np.arange(10)
y=np.sin(np.array([0,0,0,0,0,1,1,1,1,1]))

#Simple Linear Regression
model=LinearRegression()
model.fit(x.reshape(-1,1),y)
# print(model.score(x.reshape(-1,1),y))
yBest=model.predict(x.reshape(-1,1))
# # print(yBest)
# print(x.reshape(-1,1)) # jadi 2 dimensi

#Polinomial Linear Regression
model2=make_pipeline(
    PolynomialFeatures(8),
    LinearRegression()
)
model2.fit(x.reshape(-1,1),y)
yBest2=model2.predict(x.reshape(-1,1))


#Lasso Regression L1 regularization +absolute value B
from sklearn.linear_model import Lasso
modelL=make_pipeline(
    PolynomialFeatures(8),
    Lasso(alpha=1e-15,normalize=True,max_iter=100000)
)
modelL.fit(x.reshape(-1,1),y)
yBestL=modelL.predict(x.reshape(-1,1))



#Ridge Regression L2 +k.Lambda

from sklearn.linear_model import Ridge
modelR=make_pipeline(
    PolynomialFeatures(8),
    Ridge(alpha=1e-15,normalize=True,max_iter=100000)
)
modelR.fit(x.reshape(-1,1),y)
yBestR=modelR.predict(x.reshape(-1,1))


plt.figure('Regression',figsize=(20,30))
plt.subplot(221)
plt.title('Simple Linear Regression')
plt.scatter(x,y,color='y')
plt.plot(np.sort(x),np.sort(yBest),'r-')

plt.subplot(222)
plt.title('Polynomial Linear Regression')
plt.scatter(x,y,color='y')
plt.plot(np.sort(x),np.sort(yBest2),'g-')

plt.subplot(223)
plt.title('Polynomial Lasso Regression')
plt.scatter(x,y,color='y')
plt.plot(np.sort(x),np.sort(yBestL),'k')

plt.subplot(224)
plt.title('Polynomial Ridge Regression')
plt.scatter(x,y,color='y')
plt.plot(np.sort(x),np.sort(yBestR),'b')
plt.show()

