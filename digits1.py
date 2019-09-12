import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.datasets import load_digits


dataDigits=load_digits()
# print(dir(dataDigits))
# print((dataDigits)['data'][0])
# print(dataDigits['images'][0])
dataFinal=pd.DataFrame(
    dataDigits['data'],
    columns=np.arange(len(dataDigits['data'][0]))
    )
dataFinal['target']=dataDigits['target']

from sklearn.model_selection import train_test_split
xtrain,xtes,ytrain,ytes=train_test_split(
    dataDigits['data'],
    dataFinal['target'],
    test_size=.1
    )

# print(len(xtrain))

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=10000)
model.fit(xtrain,ytrain)
# print(model.score(xtrain,ytrain))
# print(xtes)
# print(ytes[1757])
# print(model.predict(xtes))
ytes=list(ytes)

#visualization + prediction
# print(prediksi)
fig=plt.figure('LogReg',figsize=(10,5))
# for item in range(10):
#     akurasi=round(model.score(xtes,ytes)*100,2)
#     plt.subplot(2,5,item+1)
#     prediksi=model.predict(xtes[item].reshape(1,-1))[0]
#     plt.imshow(xtes[item].reshape(8,8),cmap='gray')
#     plt.title(
#         'P={}|D={}|A={}'.format(prediksi,ytes[item],akurasi)
#         )
# # plt.show()

from PIL import Image
import PIL.ImageOps
gbr=Image.open('download.png').convert('L')
gbr=gbr.resize((8,8))
gbr=PIL.ImageOps.invert(gbr)
gbrArr=np.array(gbr)
# print(gbrArr[0])
# plt.imshow(gbrArr,cmap='gray')
# plt.show()
prediksi=model.predict(gbrArr.reshape(1,-1))
print(prediksi)
plt.imshow(gbrArr,cmap='gray')
plt.title(
    'P={}'.format(prediksi)
    )
# plt.show()

import joblib
joblib.dump(model,'digits.joblib')

