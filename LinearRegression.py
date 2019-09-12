import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model

data=pd.read_csv('tes.csv')

plt.scatter(data['luas'],data['harga'])
# plt.show()

#=========================== Linear Regression ==============================
# y=mX+c
# Model ML metode linear regression
model=linear_model.LinearRegression()

# Training Model dg data yg kita punya
# model.fit(dependent,independent)
model.fit(data[['luas']],data['harga'])

m=model.coef_
c=model.intercept_
# print(m[0])
# print(c)

# Prediksi model.predict([[2D]])
# print(model.predict([[100]]))
# print(model.predict([[3000]]))
print(model.predict(data[['luas']]))


plt.style.use('ggplot')
plt.plot(
    data['luas'],data['harga'],'ro',
    data['luas'],model.predict(data[['luas']]),'g-'
    )
plt.grid(True)
plt.xlabel('Luas(m2)')
plt.ylabel('Harga(Rp)')
plt.legend(['Data','Best Fit Line'])
plt.show()