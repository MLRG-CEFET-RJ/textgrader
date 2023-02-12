def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score, StratifiedKFold
from sklearn.metrics import cohen_kappa_score,r2_score,make_scorer,mean_squared_error
from dags.model_training.ordinalClassifier import *

import pandas as pd
import numpy as np
import pickle

from dags.utils import * 

 

class PipelineModel():

    def __init__(self):
        self.predicted_variable = 'score'

        long_param_grid    = {'min_samples_leaf': [8],"max_depth": [6,8]}
        short_param_grid = {'min_samples_leaf': [8],"max_depth": [6]}
    
        self.param_grid = short_param_grid


    def split_data(self,df):
        df = df.drop(columns = ["domain","text","new_text"],errors = "ignore")
        id = df [["text_id","text_set"]]
        y = df[[self.predicted_variable]]
        x = df.drop(columns = ["text_id","text_set"] + [self.predicted_variable] , errors = "ignore")

        return x,y,id

    def optimize_and_fit(self,df):

        sk = StratifiedKFold(n_splits= 5)

        X,y,id = self.split_data(df)
        gs = GridSearchCV(estimator = self.estimator, param_grid = self.param_grid, cv = sk)
        model = gs.fit(X,y)

        return model 
        

class PipelineRegressor(PipelineModel):
    def __init__(self):
        super().__init__()
        self.estimator = RandomForestRegressor()
        self.scorer = None

class PipelineClassifier(PipelineModel):
    def __init__(self):
        super().__init__()
        self.estimator = RandomForestClassifier()
        self.scorer = make_scorer(cohen_kappa_score,weights = 'quadratic')

       
class PipelineOrdinalClassifier(PipelineModel):
    def __init__(self):
        super().__init__()
    
  
        self.estimator = ordinalClassifier()

  
    def optimize_and_fit(self,df):
        X,y,id = self.split_data(df)
        model = self.estimator.fit(X,y)

        return model
    
        


 









