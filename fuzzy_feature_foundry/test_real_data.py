from fuzzy_features import * 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer, 
TfidfVectorizer) 
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    cosine_distances, 
    manhattan_distances, 
    euclidean_distances
)
from functools import partial
if __name__ == '__main__' : 

    data_1 = pd.read_csv('././claims.csv')
    data_1 = data_1[data_1['country']=='Chile'].reset_index()
    data_1['eta_pod'] = pd.to_datetime(data_1['eta_pod'])
    data_2 = pd.read_csv('././shipments.csv')
    data_2 = data_2[data_2['conveyance']=='ocean'].reset_index()
    data_2['departure_date'] = pd.to_datetime(data_2['departure_date'])
    data_1['eta_pod'] = data_1['eta_pod'].view(int) / 10**9
    data_2['departure_date'] = data_2['departure_date'].view(int) / 10**9
    fz = FuzzyCombiner(data_1, data_2)
    
    f = partial(tfidf_cosine,n_neighbors=10,
        analyzer='char', ngram_range=(3,3), max_features=5000)
    fz.compare('client_exporter','client_exporter', 'client',f)
    print("1")
    fz.compare('product','product','product',dummies_cosine)
    print("2")
    fz.compare('eta_pod','departure_date','date',continuous_nearest_neighbors)
    print(fz.evals['product'])
    print(fz.evals['client'])
    print(fz.evals['date'])
    df = fz.compile_evaluations()
    print(df)
    #c = continuous_nearest_neighbors(data_1['eta_pod'],data_2['departure_date'])
    #print(c)