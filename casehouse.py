import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model

data=pd.read_csv('housing.csv')


data.dropna()
# print(data.iloc[0])
# print(median_income)
# for item in data['housing_median_age']:
    # print(item)


# print(location)
# ============Cek Korelasi=================
# corr=data.corr()
# print(corr)
# # sb.heatmap(corr)
# plt.show()
# plt.imshow(corr,cmap='hot_r')
# plt.colorbar()
# plt.show()


# =======================Multivariate Linear Regression==========================
model=linear_model.LinearRegression()
model.fit(data[['housing_median_age','total_rooms','median_income']],data['median_house_value'])

print(model.coef_)
print(model.intercept_)

print(model.predict([[41,880,8.3252]]))