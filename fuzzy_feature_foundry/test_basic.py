from fuzzy_features import * 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (CountVectorizer, 
TfidfVectorizer) 
from datetime import datetime
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    cosine_distances, 
    manhattan_distances, 
    euclidean_distances
)
from sklearn.neighbors import NearestNeighbors
from functools import partial
data_1 = pd.DataFrame(
    {
        'c_1':[
            'export pro',
            'fruit export sa',
            'king of fruit exp'
        ],
        'c_2':[
            'blueberry',
            'blueberry',
            'cherry'
        ],
        'c_3':[
            datetime(2022, 4, 15),
            datetime(2023, 2, 25),
            datetime(2022, 10, 8)
        ]
    }
)
data_2 = pd.DataFrame(
    {
        'client':[
            'export professionals',
            'export professionals',
            'fruit logistics',
            'fruit export',
            'king fruit exports'
        ],
        'product':[
            'blueberry',
            'cherry',
            'blueberry',
            'banana',
            'cherry'
        ],
        'rand_field':[
            10,
            15,
            3,
            19,
            1
        ],
        'date':[
            datetime(2022, 4, 19),
            datetime(2021,12,29),
            datetime(2023, 2, 20),
            datetime(2023,1,11),
            datetime(2022, 10, 9)
        ]
    }
)

fz = FuzzyCombiner(
    data_1, 
    data_2, 
    data_1_name='claim',
    data_2_name='ship')
f = partial(tfidf_cosine,n_neighbors=2,
analyzer='char', ngram_range=(3,3), max_features=5000)
fz.compare('c_1','client','client',f)
fz.compare('c_2','product','product',dummies_cosine)
print(fz.evals['client'])
print(fz.evals['product'])
