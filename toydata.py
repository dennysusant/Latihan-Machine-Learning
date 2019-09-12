import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.datasets import load_boston

# ['DESCR', 'data', 'feature_names', 'filename', 'target']
# .shape(jumlah data)
# ===================================Load Data===============================
dataBoston=load_boston()
# print(dataBoston.keys())
# print(dir(dataBoston))
# print(dataBoston['data'].shape)
# print(dataBoston['data'][0])
# print(dataBoston['feature_names']) #nama kolom
# print(dataBoston['target'])

df=pd.DataFrame(
    dataBoston['data'],
    columns=dataBoston['feature_names']
)
print(df)
# ==========================Cek Korelasi=====================================
corr=df.corr()
corr

# sb.heatmap(corr)
# plt.show()

# ==============================Regresi===============================
model=linear_model.LinearRegression()
model.fit(df[['ZN','CHAS','RM','DIS','B']],df['price'])

print(model.coef_)
print(model.intercept_)

model.predict(df.head(1)[['ZN','CHAS','RM','DIS','B']])