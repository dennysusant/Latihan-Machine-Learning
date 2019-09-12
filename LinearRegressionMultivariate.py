import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model


data={
    'luas':[2500,3000,3200,3600,4000],
    'kamar':[2,3,3,2,4],
    'usia':[10,15,20,18,8],
    'harga':[500,550,620,600,720]
}
data=pd.DataFrame(data)
# ============Cek Korelasi=================
corr=data.corr()
# print(corr)
sb.heatmap(corr)
plt.show()
# =======================Multivariate Linear Regression==========================
# y=m1x1+m2x2+m3x3+c

# model=linear_model.LinearRegression()
# model.fit(data[['luas','kamar','usia']],data['harga'])

# # print(model.coef_)
# # print(model.intercept_)

# print(model.predict([[1000,5,1]]))