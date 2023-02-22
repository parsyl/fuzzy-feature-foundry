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

def vectorizer_distance(
    data=None,
    vectorizer=TfidfVectorizer,
    distance=cosine,
    **vec_kwargs
) : 
    vectorizer = vectorizer(**vec_kwargs)
    vec_text = vectorizer.fit_transform(data)



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
                    'data_types' : []
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
        n_1, n_2 = len(data_1), len(data_2)
        self.data_1_ids = [*range(n_1)]
        self.data_2_ids = [*range(n_1, n_1+n_2)]
        
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

    def add_field_comparisons(self, field_1, field_2) : 
        dtype = self._validate_field_matches(field_1, field_2)
        addition = (field_1, field_2)

        if 'pairwise_matches' not in  self.config : 
            self.config['pairwise_matches'] = {
                'columns':[],
                'data_types':[]
            }
        self.config['pairwise_matches']['columns'].append(addition) 
        self.config['pairwise_matches']['data_types'].append(dtype)
    
    def _able_to_add_eval(self, field_1, field_2) : 
        assert (field_1, field_2) in \
            self.config['pairwise_matches']['columns'], \
                f"{field_1}, {field_2} not added to matches"
    
    def add_evaluation(
        self, 
        field_1, 
        field_2, 
        eval_function,
        **kwargs
    ) : 
        self._able_to_add_eval(field_1, field_2)
        self.config['comparison_metrics'][(field_1,field_2)] = \
            eval_function
        
        evaluation_data = np.concatenate(
            (
                self.data_1[field_1].values,
                self.data_2[field_2].values
            )
        )
        eval = eval_function(evaluation_data, **kwargs)
        out = {(d_1, d_2) : eval[d_1, d_2]
        for d_1 in self.data_1_ids 
        for d_2 in self.data_2_ids
        }
        self.evals[(field_1, field_2)] = out
    
    def compile_evaluations(self) : 
        """
        
        """
        #make sure able to compile, perform some check
        evals = []
        nms = self.config['names'][0]
        for k, v in self.evals.items() : 
            e = pd.DataFrame().from_dict(v, orient='index')
            nm = '_'.join([nms[0],k[0],nms[1],k[1]])
            e.columns = [nm]
            evals.append(e)
        return pd.concat(evals, axis=1)




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
    
    """
    matrix = pd.get_dummies(data, **kwargs).to_numpy()
    distance = distance_func(matrix, matrix)
    return distance
def dummies_cosine(data, **kwargs) : 
    """
    
    """
    return dummies_met(
        data,
        cosine_distances,
        **kwargs
    )
