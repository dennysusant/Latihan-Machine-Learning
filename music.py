import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


data=pd.read_csv('song.csv')
# print(data.columns)
df=data[['TITLE','YEAR','THEME','ARTIST',]]
df=df.dropna()

def kombinasi(i):
    return str(i['THEME'])+ '$' +str(i['ARTIST'])
df['x']=df.apply(kombinasi,axis=1)


cov=CountVectorizer(
    tokenizer=lambda data: data.split('$')
)

datatheme=cov.fit_transform(df['x'])
# print(cov.get_feature_names())
# # print(datatheme)
from sklearn.metrics.pairwise import cosine_similarity 
skorTheme=cosine_similarity(datatheme)
# # print(skorTheme)
# #tes by THEME
suka='Eight Days a Week'
indexsuka=df[df['TITLE']==suka].index.values[0]
musicscore=list(enumerate(skorTheme[indexsuka]))
sortmusic=sorted(musicscore,key=lambda i:i[1],reverse=True)
for item in sortmusic[:10]:
    if df.iloc[item[0]]['TITLE'] !=suka:
        judul=df.iloc[item[0]]['TITLE']
        artist=df.iloc[item[0]]['ARTIST']
        print(f'{judul} ({artist}) {round(item[1]*100)}%')
