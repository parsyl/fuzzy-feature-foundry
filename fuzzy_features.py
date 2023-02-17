import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import (CountVectorizer, 
TfidfVectorizer)

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

        
        
    def _validate_field_matches(self, field_1, field_2) : 
        """
        hidden function to validate the fields will work together
        fields must be same type or castable to same type in order
        to work.
        """
        assert field_1 in self.data_1.columns
        assert field_2 in self.data_2.columns
        assert self.data_1[field_1].dtypes == self.data_2[field_2].dtypes
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
            self.config['pairwise_matches']['columns']
    def add_evaluation(self, field_1, field_2, eval_function) : 
        self._able_to_add_eval(field_1, field_2)
        self.config['comparison_metrics'][(field_1,field_2)] = \
            eval_function


        


