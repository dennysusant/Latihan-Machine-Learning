import pickle 
import joblib


# #Load dengan Pickle
# with open ('modelPickle.pkl','rb') as modelku:
#     modelLoad=pickle.load(modelku)


# Load dengan JOBLIB
modelLoad=joblib.load('modelJoblib.joblib')
hasil=modelLoad.predict([[2.875000,15.000000,5.891892,1.124324,960.000000,2.594595,35.160000,-117.990000]])
print(hasil)