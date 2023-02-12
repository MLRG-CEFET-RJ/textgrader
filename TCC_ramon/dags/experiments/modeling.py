from msilib.schema import Directory
from dags.utils import * 
from dags.model_training.train_model import *
from sklearn.metrics import cohen_kappa_score
 
import os 


## desabilita logs dos métodos importados anteriormente
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})
## fecha por aqui


import pandas as pd
from dags.predict.predict import *
from dags import config

import logging 
 

logger = logging.getLogger(__name__)
print(__name__)
logger.setLevel(config.LOGLEVEL)


class model_trainer():

    def __init__(self):
        self.topic_numbers = [10,20,30,40,50,100]
        self.d2v_vector_sizes =  [512,256,128,64,32]
        self.tf_idf_vector_sizes =  [512,256,128,64,32]


    def predict_with_model(self,method,version,set,test_df,predicted_variable = "score"):

        model_path = os.path.join(self.model_directory,f'set_{set}','domain_1',method,f'version_{version}')
        model_filename = f'rf_model.pkl'


        model = get_model_from_pickle(model_path,model_filename)
    
        X = test_df[model.feature_names_in_]

        y_pred = model.predict(X)
        y_pred = np.round(y_pred)

        test_df["prediction"] = y_pred

        score = cohen_kappa_score(test_df["prediction"],test_df[predicted_variable], weights = "quadratic")
        
        return test_df , score
        
    def save_model(self,model,output_path):
        output_filename = f'rf_model.pkl'
        save_as_pickle(model,output_path,output_filename)
         
    
    def experiments(self):
        dicio = { 'USE':['unique'],
                  'LSI':[10,20,30,40,50,100],
                  'DOC_2_VEC':[32,64,128,256,512],
                  'TF_IDF':[32,64,128,256,512]}

        logging.info("starting to train models")

        for method in dicio.keys():
            versoes = dicio[method]
            for version in versoes:
                logging.info(f"training models with the methpd {method} version {version} dataset")
                for i in self.text_range:
                    input_directory = os.path.join(self.selected_container,'processed','train',f'set_{i}','domain_1',method,f'version_{version}')   
                    input_filename = f'features.parquet'
                    
                    logging.info(f"text {i}")
                
                    df = pd.read_parquet(os.path.join(input_directory,input_filename))
                    df = df.dropna()
                    
                    model = self.pipeline.optimize_and_fit(df)
                    
                    output_path = os.path.join(self.model_directory,f'set_{i}','domain_1',method,f'version_{version}')
                 
                    
                    self.save_model(model,output_path)
        logging.info("finished training models")


    def evaluate(self):
        dicio = { 'USE':['unique'],
                  'LSI':[10,20,30,40,50,100],
                  'DOC_2_VEC':[32,64,128,256,512],
                  'TF_IDF':[32,64,128,256,512]}

        logging.info("starting to evaluate models")

        for method in dicio.keys():
            versoes = dicio[method]

            listao = []
 
            for i in self.text_range:
                score_list = []
                
                for version in versoes:
                    logger.info(f"evaluating models with the methpd {method} version {version} dataset")

                    logger.info(f'text {i}')

                    input_directory = os.path.join(self.selected_container,'processed','test',f'set_{i}','domain_1',method,f'version_{version}')  
                    input_filename = f'features.parquet'

                
                    output_directory = os.path.join(self.selected_container,'predictions','test',self.model_type,f'set_{i}','domain_1',method,f'version_{version}')  
                    output_filename = f'predictions.parquet'

                    test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
                
                    predictions_df,score = self.predict_with_model(method,version,i,test_df)

                    score_list.append(score)

                    save_parquet(predictions_df,output_directory,output_filename)

                tupla = tuple([i] + score_list)
                listao.append(tupla)

            df_results = pd.DataFrame.from_records(listao)
            df_results.columns = ['text_set'] + [f'version_{i}' for i in versoes]
 
            EXPERIMENT_RESULT_FOLDER = os.path.join(self.selected_container,'predictions','experiment_results',self.model_type)

            save_parquet(df_results,EXPERIMENT_RESULT_FOLDER,f'{method}_results.parquet')

            logging.info(f'finished evaluating {method} models')


    
class regression_model_trainer(model_trainer):
    def __init__(self,selected_container,text_range):
        super().__init__()
       
        self.selected_container = selected_container
        self.text_range = text_range
        self.input_directory = os.path.join(self.selected_container,'processed','train') 
        self.model_type = 'regressor'
        self.model_directory = os.path.join(self.selected_container,'model',self.model_type)  
        self.pipeline = PipelineRegressor()

         

class classification_model_trainer(model_trainer):
    def __init__(self,selected_container,text_range):
        super().__init__()
              
        self.selected_container = selected_container
        self.text_range = text_range
        self.input_directory = os.path.join(self.selected_container,'processed','train')   
        self.model_type = 'classifier'
        self.model_directory = os.path.join(self.selected_container,'model',self.model_type)  

        self.pipeline = PipelineClassifier()

    

class ordinal_classification_model_trainer(model_trainer):
    def __init__(self,selected_container,text_range):
        super().__init__()
              
        self.selected_container = selected_container
        self.text_range = text_range
        self.input_directory = os.path.join(self.selected_container,'processed','train')   
        self.model_type = 'ordinal_classifier'
        self.model_directory = os.path.join(self.selected_container,'model',self.model_type)  
        self.pipeline = PipelineOrdinalClassifier()

    def save_model(self,model_dict,output_path):
        dicio_apontamentos = {}
        for key,value in model_dict.items():
            output_filename = f'rf_model_bigger_than_{key}.pkl'
            save_as_pickle(value,output_path,output_filename)

            dicio_apontamentos[key] = output_filename
        save_as_pickle(dicio_apontamentos,output_path,'model_dictionary.pkl')

    def predict_with_model(self,method,version,set,test_df,predicted_variable = "score"):

        model_path = os.path.join(self.model_directory,f'set_{set}','domain_1',method,f'version_{version}')

        model_dicio = self.load_models(model_path)
 
        algorithm = ordinalClassifier()
    
        y_pred = algorithm.predict(test_df,model_dicio = model_dicio)
        y_pred = np.round(y_pred)

        test_df["prediction"] = y_pred

        score = cohen_kappa_score(test_df["prediction"],test_df[predicted_variable], weights = "quadratic")
        
        return test_df , score
         
    def load_models(self,model_path):
        dicio = get_model_from_pickle(model_path,'model_dictionary.pkl')

        new_dicio = {}

        for key,name in dicio.items():
            modelo = get_model_from_pickle(model_path,name)
            new_dicio[key] = modelo

        return new_dicio
