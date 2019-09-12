import pandas as pd
import sqlalchemy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn import linear_model


df=pd.read_excel('indo.xls',header=3,skipfooter=2,na_values=['-'])
df.rename(columns={'Unnamed: 0':'Provinsi'},inplace=True)
# df=df.fillna(method='bfill',axis=1)

df=df.dropna()
# print(df)



# corr=df.corr()
# print(corr)
# sb.heatmap(corr)
# plt.show()

model=linear_model.LinearRegression()
model.fit(df[[1971,1980,1990,1995,2000]],df[2010])


print(model.coef_)
print(model.intercept_)

print(model.predict([[119208229.0,147490298.0,179378946.0,194754808.0,206264595.0]]))