#collaborative=corr
import numpy as np 
import pandas as pd 

data=pd.read_csv('z.csv')
corrMatrix=data.corr(method='pearson')
#pearson method : standard corr coefficient
#kendall method : kendall Tau corr coeff
#spearman method : Spearman rank correlation
# print(corrMatrix)

#case
fafa=[('kartun1',5),('drama3',1)]
dfSkor=pd.DataFrame()
for film,rating in fafa:
    skor=corrMatrix[film]*rating
    skor=skor.sort_values(ascending=False)
    dfSkor=dfSkor.append(skor)
totalSkor=dfSkor.sum().sort_values(ascending=False)
print(totalSkor[totalSkor>0])