import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer


data=pd.read_csv('movie.csv')
# print(data.isnull().sum())
# print(data['genres'][0].split('|'))
# feature:title & genre
df=data[['title','genres']]
df=df.iloc[:8000]
# print(df)
# print(df['genres'])

cov=CountVectorizer(
    tokenizer=lambda i: i.split('|')
)
datamx=cov.fit_transform(df['genres'])
print(datamx)

from sklearn.metrics.pairwise import cosine_similarity 
skorKesamaan=cosine_similarity(datamx)
# print(skorKesamaan)

#test
suka='Toy Story (1995)'
indexsuka=df[df['title']==suka].index.values[0]
filmscore=list(enumerate(skorKesamaan[indexsuka]))
sortfilm=sorted(filmscore,key=lambda i:i[1],reverse=True)
# print(sortfilm)

#10film rekomendasi
# print(sortfilm[:10])
for item in sortfilm[:10]:
    print(df.iloc[item[0]]['title'],item[1])

# # print(cov.get_feature_names())
# # print(datamx.toarray()[1])




# from sklearn.metrics.pairwise import cosine_similarity 
# skorKesamaan=cosine_similarity(datamx)
# print(skorKesamaan)