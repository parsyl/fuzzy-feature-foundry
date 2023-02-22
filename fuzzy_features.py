import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
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



# how to make config file?
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
    ) : 
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
        
        evaluation_data = np.concatenate(
            (
                self.data_1[field_1].values,
                self.data_2[field_2].values
            )
        )
        eval_matrix = eval_function(evaluation_data, **kwargs) 
        out = {(d_1, d_2) : [
            self.data_1.loc[d_1,field_1], 
            self.data_2.loc[d_2,field_2], 
            eval_matrix[d_1, d_2+self.n_1] # note - assumes index of data is range index
        ]
        for d_1 in self.data_1_ids 
        for d_2 in self.data_2_ids
        }
        nms = self.config['names'][0]
        cols = [
            nms[0]+'_'+field_1, 
            nms[1]+'_'+field_2, 
            '_'.join([nms[0],field_1,nms[1],field_2])
        ]
        self.evals[(field_1, field_2)] = pd.DataFrame().\
            from_dict(out, orient='index',columns=cols)
    
    def compile_evaluations(self) : 
        """
        Bring all evaluations into a single dataframe
        """
        #make sure able to compile, perform some check
        evals = [v for k, v in self.evals.items()]
        self.xcompare_data = pd.concat(evals, axis=1)
        return self.xcompare_data




def vec_met_distance(data, vectorizer, distance_func, **kwargs) : 
    """
    generic function for vectorizing text and comparing similarity
    data - 1d array of strings
    vectorizer - a sklearn text vectorizer
    distance - an sklearn distance metric function
    kwargs - keyword args for vectorizer
    """
    vec = vectorizer(**kwargs)
    matrix = vec.fit_transform(data)
    distance = distance_func(matrix, matrix)
    return distance

def tfidf_cosine(data, **kwargs) : 
    """
    perform tfidf/cosine similarity distance on a data vector
    data - 1d array of strings
    """
    return vec_met_distance(
        data,
        TfidfVectorizer,
        cosine_distances,
        **kwargs
        )
def count_cosine(data, **kwargs) : 
    """
    perform countvec/cosine similarity distance on a data vector
    data - 1d array of strings
    """  
    return vec_met_distance(
        data,
        CountVectorizer,
        cosine_distances,
        **kwargs
        )      

def dummies_met(data, distance_func, **kwargs) : 
    """
    matrix is one-hot encoding of column
    """
    matrix = pd.get_dummies(data, **kwargs).to_numpy()
    distance = distance_func(matrix, matrix)
    return distance
def dummies_cosine(data, **kwargs) : 
    """
    apply cosine distance to one-hot matrix
    """
    return dummies_met(
        data,
        cosine_distances,
        **kwargs
    )
    
def continuous_distance(data, **kwargs) : 
    """
    cross compare continuous field
    """
    i,j = np.meshgrid(data, data, indexing='ij')
    return np.subtract(i,j)

def abs_continuous_distance(data, **kwargs) : 
    """
    `continous_distance` but positive-negative indifferent
    """
    return np.abs(continuous_distance(data, **kwargs))

def date_difference(data, **kwargs) : 
    """
    distance of date field, scaled by 
    """
    pass

def abs_date_difference(data, **kwargs) : 
    """
    absolute difference in dates (positive-negative indifference)
    """
    pass

