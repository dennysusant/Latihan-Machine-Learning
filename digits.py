import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import linear_model
from sklearn.datasets import load_digits


dataDigits=load_digits()


# print(dir(dataDigits))
# print(dataDigits['target_names'])
dataFinal=pd.DataFrame(
    dataDigits['data'],
    columns=np.arange(len(dataDigits['data'][0]))
    )
dataFinal['target']=dataDigits['target']


#Splitting
from sklearn.model_selection import train_test_split
xtrain,xtes,ytrain,ytes=train_test_split(
    dataDigits['data'],
    dataFinal['target'],
    test_size=.05
    )


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='lbfgs')
model.fit(xtrain,ytrain)
# print(model.score(xtrain,ytrain))
# print(xtes)
# print(ytes)
# print(model.predict(xtes))
# print(dataFinal)
# for item in range (0,len(angka)):
#     plt.subplot(1,len(angka),item+1)
#     plt.imshow(dataDigits['images'][int(angka[item])])
#     plt.title('Ini {}'.format(dataDigits['target'][int(angka[item])]))
#     plt.xticks(color='w')
#     plt.yticks(color='w')
# plt.show()
import joblib
joblib.dump(model,'digits.joblib')

# import pickle 
# with open ('modelPickle.pkl','wb') as modelku:
#     pickle.dump(model,modelku)


