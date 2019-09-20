import numpy as np 
import pandas as pd 


x=np.sort(np.random.randn(100))
y=np.sin(-x) 
# +np.random.rand(100)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
modelLR=LinearRegression()
modelLR.fit(x.reshape(-1,1),y)
modelDTR=DecisionTreeRegressor()
modelDTR.fit(x.reshape(-1,1),y)
modelDTR5=DecisionTreeRegressor(max_depth=5)
modelDTR5.fit(x.reshape(-1,1),y)

#Predict
yLR=modelLR.predict(x.reshape(-1,1))
yDTR=modelDTR.predict(x.reshape(-1,1))
yDTR5=modelDTR5.predict(x.reshape(-1,1))

import matplotlib.pyplot as plt 
plt.plot(
    x,y, 'yo',
    x,yLR,'g-',
    x,yDTR,'b-',
    x,yDTR5,'k-'
    )
plt.show()