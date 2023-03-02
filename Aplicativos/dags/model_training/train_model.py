from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import cohen_kappa_score,r2_score,make_scorer,mean_squared_error

import pandas as pd
import numpy as np
import pickle

from dags.utils import * 



def split_data(df,predicted_variable):
    df = df.drop(columns = ["domain","text","new_text"],errors = "ignore")
    id = df [["text_id","text_set"]]
    y = df[[predicted_variable]]
    x = df.drop(columns = ["text_id","text_set"] + [predicted_variable] , errors = "ignore")

    return x,y,id

def pipeline_random_forest(df,predicted_variable = "score"):
  
    X,y,id = split_data(df,predicted_variable = predicted_variable)

    rf = RandomForestRegressor()

    long_param_grid = {'min_samples_leaf': [8,10,12],"max_depth": [6,8,10]}
    short_param_grid = {'min_samples_leaf': [8],"max_depth": [6]}
    param_grid = short_param_grid
    gs = GridSearchCV(estimator = rf, param_grid = param_grid)

    model = gs.fit(X,y)

    return model 















