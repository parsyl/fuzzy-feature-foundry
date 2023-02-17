from hologram import T
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
    strings = ['a','b','c','d']
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
    def eval_func (data) : 
        tfidf = TfidfVectorizer()
        matrix = tfidf.fit_transform(data)
        
    fz = FuzzyCombiner(make_data(),make_data())
    fz.add_field_comparisons('col_0','col_0')
    fz.add_field_comparisons('col_1','col_1')
    fz.add_evaluation('col_0','col_0', cosine)
    print(fz.config)

