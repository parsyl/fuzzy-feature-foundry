import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer, 
    TfidfVectorizer
    )
from sklearn.metrics.pairwise import (
    cosine_similarity, 
    cosine_distances, 
    manhattan_distances, 
    euclidean_distances
)
from functools import partial, reduce
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from typing import Literal, List, Union, Tuple

# how to make config file?
### todo Change this to distinct & lookup
### use sklearn nearest neighbors function instead
class FuzzyCombiner() : 
    """
    This class combines two datasets and cross-compares 
    the fields of interest by the metric/methodology of
    interest. It outputs a dataframe with comparisons
    for each potential pair so the user can build a model
    for matches
    """
    def __init__(
            self, 
            data_1=pd.DataFrame, 
            data_2=pd.DataFrame,
            config_file=None,
            data_1_name=None,
            data_2_name=None
        ) : #typing, kwargs?
        self.data_1, self.data_2 = data_1, data_2
        self.field_matches = []
        if config_file == None :
            self.config = {
                'pairwise_matches' : {
                    'columns' : [],
                    'data_types' : [],
                    'names':[]
                },
                'comparison_metrics' : {} ,
                'names': []
            }
        name_1 = data_1_name if data_1_name is not None \
            else 'data_1'
        name_2 = data_2_name if data_2_name is not None \
            else 'data_2'
        
        self.config['names'].append((name_1, name_2))
        self.evals = {}
        self.n_1, self.n_2 = len(data_1), len(data_2)
        self.data_1_ids = data_1.index#[*range(n_1)] #preserve original index
        self.data_2_ids = data_2.index#[*range(n_1, n_1+n_2)] #want to preserve original index?
        
    def _validate_field_matches(self, field_1, field_2) : 
        """
        hidden function to validate the fields will work together
        fields must be same type or castable to same type in order
        to work.
        """
        assert field_1 in self.data_1.columns, f"{field_1} not in data_1 columns"
        assert field_2 in self.data_2.columns, f"{field_2} not in data_2 columns"
        assert self.data_1[field_1].dtypes == self.data_2[field_2].dtypes, "data type mismatch"
        return self.data_1[field_1].dtypes

    def add_field_comparisons(self, field_1, field_2, out_name=None) : # give it name other than tuple of fields
        dtype = self._validate_field_matches(field_1, field_2)
        addition = (field_1, field_2)
        name = out_name if out_name is not None else field_1 #default to field 1
        if 'pairwise_matches' not in  self.config : 
            self.config['pairwise_matches'] = {
                'columns':[],
                'data_types':[],
                'names':[]
            }
        self.config['pairwise_matches']['columns'].append(addition) 
        self.config['pairwise_matches']['data_types'].append(dtype)
        self.config['pairwise_matches']['names'].append(name)
    
    def _able_to_add_eval(self, field_1, field_2) : 
        assert (field_1, field_2) in \
            self.config['pairwise_matches']['columns'], \
                f"{field_1}, {field_2} not added to matches"
    
    def add_evaluation( 
        self, 
        name,
        eval_function,
        **kwargs
    ) : # add some kind of filter so it doesn't blow the thing up
        """
        Add an evaluation for comparison `name` using eval function
        and keyword args `kwargs`
        """
        names_list = self.config['pairwise_matches']['names']
        pairs_list = self.config['pairwise_matches']['columns']
        assert name in names_list, f'not a valid name : {name}'
        field_dict = dict(zip(names_list, pairs_list))
        field_1, field_2 = field_dict[name]
        self._able_to_add_eval(field_1, field_2)
        self.config['comparison_metrics'][name] = \
            eval_function
        
        data_1 = self.data_1[field_1]
        data_2 = self.data_2[field_2]

        
        eval_matrix = eval_function(data_1, data_2, **kwargs) 
        out = {(d_1, d_2) : [
            self.data_1.loc[d_1,field_1], 
            self.data_2.loc[d_2,field_2], 
            eval_matrix[(self.data_1.loc[d_1,field_1], 
                self.data_2.loc[d_2,field_2])] 
        ]
        for d_1 in self.data_1_ids 
        for d_2 in self.data_2_ids
        if (self.data_1.loc[d_1,field_1], 
                self.data_2.loc[d_2,field_2]) in eval_matrix
        }
        nms = self.config['names'][0]
        cols = [
            nms[0]+'_'+field_1, 
            nms[1]+'_'+field_2, 
            '_'.join([nms[0],field_1,nms[1],field_2])
        ]
        self.evals[name] = pd.DataFrame().\
            from_dict(out, orient='index',columns=cols)
    def compare(
        self,
        field_1,
        field_2,
        out_name,
        eval_function,
        **kwargs
    ) : 
        """
        compare
        """
        self.add_field_comparisons(field_1, field_2, out_name)
        self.add_evaluation(out_name, eval_function, **kwargs)
    
    # function to filter evaluation
    def compile_evaluations(self) : # add some kind of filter for each
        """
        Bring all evaluations into a single dataframe
        """
        #make sure able to compile, perform some check
        evals = [v for k, v in self.evals.items()]
        self.xcompare_data = pd.concat(evals, axis=1, join='inner')
        return self.xcompare_data



####### These functions are what you will use
def preprocess_unique(data_1:pd.Series, data_2:pd.Series) : 
    """
    Get unique values in data_1 and data_2, as well
    as the unique values of the combined arrays
    """

    d_1, d_2 = data_1.unique(), data_2.unique()
    d_all = np.unique(
        np.concatenate(
            (
                d_1,
                d_2
            )
        )
    )
    return d_1, d_2, d_all

def make_comparisons(dist, neighbors, d_1, d_2) : 
    """
    can you speed this up with dict comprehension or 
    use something else?

    Given a set of values from preprocess_unique and the output
    of a nearest neighbors algorithm, store the pairs in a 
    dictionary

    dist,neighbors - output from NearestNeighbors().kneighbors(...)
    d_1 - numpy array of unique values in data_1
    d_2 - numpy array of unique values in data_2
    """
    comparisons = {}
    for i in range(dist.shape[0]) : 
        row_d , row_n = dist[i,:], neighbors[i,:]
        for d,n in zip(row_d, row_n) :
            comparisons[(d_1[i],d_2[n])] = d
    return comparisons
#    { ### would this work?
#        (d_1[i],d_2[n]) : d
#        for i in range(dist.shape[0])
#            for d,n in zip(dist[i,:], neighbors[i,:])
#    }
def dummies_met(
    data_1 : pd.Series, 
    data_2 : pd.Series, 
    distance_func : Literal['cityblock', 'cosine', 'euclidean', 
                            'l1', 'l2', 'manhattan'], 
    n_neighbors=None,
    **kwargs
    ) : 
    """
    Onehot encode a column and apply a distance function
    
    data_1 - column from dataset 1
    data_2 - column from dataset 2
    distance func - string specifying which distance function to use
    n_neighbors - number of neighbors wanted in output
    **kwargs - keyword arguments to feed into OneHotEncoder
    """
    assert distance_func in ['cityblock', 'cosine', 'euclidean', 
                            'l1', 'l2', 'manhattan']
    d_1, d_2, d_all = preprocess_unique(data_1, data_2)
    enc = OneHotEncoder(**kwargs)
    x_all = enc.fit_transform(d_all.reshape(-1,1))
    x_1, x_2 = enc.transform(d_1.reshape(-1,1)), enc.transform(d_2.reshape(-1,1))
    
    if n_neighbors is None :
        n = min(x_1.shape[0],x_2.shape[0],25)
    else : 
        n = n_neighbors
    neigh = NearestNeighbors(n_neighbors=n, 
        metric=distance_func)
    neigh.fit(x_2)
    dist, neighbors = neigh.kneighbors(x_1, 
        return_distance=True)

    return make_comparisons(dist,neighbors, d_1, d_2)

def dummies_cosine(data_1, data_2, **kwargs) : 
    """
    apply cosine distance to one-hot matrix
    """
    return dummies_met(
        data_1,
        data_2,
        'cosine',
        **kwargs
    )



def vec_nearest_neighbors(
    data_1:pd.Series,
    data_2:pd.Series,
    vectorizer,
    n_neighbors=10,
    metric : Literal['cityblock', 'cosine', 'euclidean', 
                    'l1', 'l2', 'manhattan'] = 'cosine',
    **kwargs
    ) :
    """
    Vectorize a column (text) and apply distance function

    data_1,data_2 - columns to compare
    vectorizer - a type of vectorizer
    """
    assert metric in ['cityblock', 'cosine', 'euclidean', 
                    'l1', 'l2', 'manhattan'], \
    f"{metric} is an invalid metric"
    vec = vectorizer(**kwargs)
    d_1, d_2, d_all = preprocess_unique(data_1, data_2)
    x_all = vec.fit_transform(d_all)
    x_1 = vec.transform(d_1)
    x_2 = vec.transform(d_2)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, 
        metric=metric)
    neigh.fit(x_2)
    dist, neighbors = neigh.kneighbors(x_1, 
        return_distance=True)
    comparisons = make_comparisons(dist, neighbors, d_1, d_2)
    
    
    #return pd.DataFrame().from_dict(comparisons, 
    #    orient='index',
    #    columns=['metric'])
    return comparisons

def tfidf_cosine(
    data_1, 
    data_2,
    n_neighbors=10,
    **kwargs
    ) :
    return vec_nearest_neighbors(
        data_1,
        data_2,
        vectorizer=TfidfVectorizer,
        n_neighbors=n_neighbors,
        metric='cosine',
        **kwargs
    )



def continuous_nearest_neighbors(
    data_1,
    data_2,
    n_neighbors=15,
    metric='euclidean',
    **kwargs
) : 
    """
    Nearest Neighbors on a column pair that does not need to be vectorized
    
    """
    assert True #make sure continuous columns
    d_1, d_2, d_all = preprocess_unique(data_1, data_2)
    x_1, x_2, x_all = d_1.reshape(-1,1), d_2.reshape(-1,1), d_all.reshape(-1,1)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, 
        metric=metric)
    neigh.fit(x_2)
    dist, neighbors = neigh.kneighbors(x_1, 
        return_distance=True)
    comparisons = make_comparisons(dist, neighbors, d_1, d_2)
    return comparisons

