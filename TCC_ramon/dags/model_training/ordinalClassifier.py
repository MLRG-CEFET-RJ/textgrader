import pandas as pd
from dags.utils import *
from sklearn.ensemble import *
 
class ordinalClassifier():
    def __init__(self):
        self.score_column = 'score'
        self.algorithm = RandomForestClassifier()
        #self.min_score = 0
        #self.max_score = 3

        
    ## private method to be called inside fit
    def __train_models_for_score(self,x,y):
        dicio = {}
        for i in range(self.min_score,self.max_score):
            new_y = y > i
            modelo = self.algorithm.fit(x,new_y)
            dicio.update({i:modelo})
            #save_as_pickle(modelo,self.path,f'bigger_than_{i}_rf.pkl')

        return dicio
            
           
    def fit(self,x,y):
        self.min_score = int(y.min()[0])
        self.max_score = int(y.max()[0])
        print(self.min_score)
        print(self.max_score)
        model = self.__train_models_for_score(x,y)

        return model
        
 
    ## private methods to be called inside predict
    def __predict_greater_than_number_probabilities(self,x):
        preds = pd.DataFrame()

        for i,model in self.model_dict.items():
            x = x[model.feature_names_in_]
            preds[f'Pr(>{i})'] = model.predict_proba(x).T[1]

        return preds
    
    def __converting_probabilities(self,preds):
        new_preds = pd.DataFrame()

        lista = list(self.model_dict.keys())
        #print(lista)
        min_score = lista[0]
        max_score = lista[-1] + 1
        lista = lista[1:]
        #print(lista)

        for i in lista:
            new_preds[i] =  preds[f'Pr(>{(i-1)})'] - preds[f'Pr(>{i})'] 

        new_preds[min_score] = 1 - preds[f'Pr(>{(min_score)})']
        new_preds[max_score] = preds[f'Pr(>{(max_score-1)})']

        return new_preds
    
    def __extracting_point_predictions(self,new_preds):
        y_preds = new_preds.to_numpy().argmax(axis = 1) 

        return y_preds
    
    def predict(self,x,model_dicio):
        self.model_dict = model_dicio
       
        preds = self.__predict_greater_than_number_probabilities(x)
        new_preds = self.__converting_probabilities(preds)
        y_hats = self.__extracting_point_predictions(new_preds)
        return y_hats