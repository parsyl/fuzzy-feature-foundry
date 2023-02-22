from fuzzy_features import * 
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import (CountVectorizer, 
TfidfVectorizer) 
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    cosine_distances, 
    manhattan_distances, 
    euclidean_distances
)


def make_data(ncols=2, nrows=10, dtypes=None) : 
    strings = ['allow hi','bet hi','capped','dove allow']
    ints = range(10)
    floats = np.linspace(0,1,34)
    if dtypes is None : 
        dtypes = ['string']*ncols
    def _get_space(dtype) : 
        if dtype in ['string',str] :
            return strings
        elif dtype in ['int',int,np.int32, np.int64] : 
            return ints
        elif dtype in ['float',float,np.float32,np.float64] : 
            return floats
        else :
            raise ValueError('wtf')
    
    data = {
        f"col_{i}":np.random.choice(
            _get_space(d),size=nrows, replace=True
            )
            for i, d in zip(range(ncols),dtypes)
        }
    return pd.DataFrame(data)


if __name__ == '__main__' : 

    #data_1 = pd.read_csv('claims.csv')
    #data_2 = pd.read_csv('shipments.csv')
    data_1 = make_data(nrows=5)
    data_2 = make_data(nrows=15)
    print(data_1)
    print("\n"*3)
    print(data_2)
    print("\n"*3)
    fz = FuzzyCombiner(data_1,data_2)
    fz.add_field_comparisons('col_0','col_0')
    fz.add_field_comparisons('col_1','col_1')
    fz.add_evaluation('col_0','col_0', tfidf_cosine)
    fz.add_evaluation('col_1','col_1', dummies_cosine)
    print(fz.config)
    print(fz.compile_evaluations())
    
    #fz = FuzzyCombiner(data_1, data_2)
    #fz.add_field_comparisons('client_exporter','client_exporter')
    #fz.add_field_comparisons('product','product')
    #fz.add_evaluation('client_exporter','client_exporter', tfidf_cosine)
    #print(fz.config)
    #print(fz.evals[('client_exporter','client_exporter')])
    #fz.add_evaluation('product','product',dummies_cosine)
    #print(fz.evals[('product','product')])



