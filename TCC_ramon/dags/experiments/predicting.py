from msilib.schema import Directory
from dags.utils import * 
from dags.model_training.train_model import *
 
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

 
def evaluate_lsi_predictions(selected_container,text_range,topic_numbers = [10,20,30,40,50,100]):
   
    logger.info(f'evaluating LSI results')

    listao = []

    for i in text_range:
        logger.info(f'text {i}')

        score_list = []
        for topic in topic_numbers:
            logger.info(f'{topic} tópicos')

            input_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1')  
            input_filename = f'lsi_{topic}_topics.parquet'

            model_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
            model_filename = f'lsi_{topic}_topics_rf_model.pkl'
 

            output_directory = os.path.join(selected_container,'predictions','test',f'set_{i}','domain_1')  
            output_filename = f'lsi_{topic}_topics_predictions.parquet'

            test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
            model = get_model_from_pickle(model_directory,model_filename)

            predictions_df,score = predict_with_model(model,test_df)

            score_list.append(score)

            save_parquet(predictions_df,output_directory,output_filename)

        tupla = tuple([i] + score_list)
        listao.append(tupla)

        df_results = pd.DataFrame.from_records(listao)
        df_results.columns = ['text_set','10_topicos','20_topicos','30_topicos','40_topicos','50_topicos','100_topicos']

        
        EXPERIMENT_RESULT_FOLDER = os.path.join(selected_container,'predictions','experiment_results')

        save_parquet(df_results,EXPERIMENT_RESULT_FOLDER,'lsi_topics_results.parquet')

    logger.info(f'LSI results were evaluated')

        
    return df_results


def evaluate_tf_idf_predictions(selected_container,text_range):
   
    logger.info(f'evaluating TF-IDF results')

    listao = []

    for i in text_range:
        logger.info(f'text {i}')

        score_list = []
        for n_features in [32,64,128,256,512]:
            logger.info(f'{n_features} tópicos')

            input_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1')  
            input_filename = f'tf_idf_{n_features}_features.parquet'

            model_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
            model_filename = f'tf_idf_{n_features}_rf_model.pkl'


            output_directory = os.path.join(selected_container,'predictions','test',f'set_{i}','domain_1')  
            output_filename = f'tf_idf_{n_features}_predictions.parquet'

            test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
            model = get_model_from_pickle(model_directory,model_filename)

            predictions_df,score = predict_with_model(model,test_df)

            score_list.append(score)

            save_parquet(predictions_df,output_directory,output_filename)

        tupla = tuple([i] + score_list)
        listao.append(tupla)

        df_results = pd.DataFrame.from_records(listao)
        df_results.columns = ['text_set','32_features','64_features','128_features','256_features','512_features']

        
        EXPERIMENT_RESULT_FOLDER = os.path.join(selected_container,'predictions','experiment_results')

        save_parquet(df_results,EXPERIMENT_RESULT_FOLDER,'tf_idf_results.parquet')

    logger.info(f'TF-IDF results were evaluated')

        
    return df_results


 
 



def evaluate_universal_sentence_encoder_predictions(selected_container,text_range):

    logger.info(f'evaluating universal sentence encoder results')


    listao = []

    for i in text_range:

        logger.info(f'text {i}')

        score_list = []
  
        input_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1')  
        input_filename = f'universal_sentence_encoder.parquet'

        model_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
        model_filename = f'universal_sentence_encoder_rf_model.pkl'

        output_directory = os.path.join(selected_container,'predictions','test',f'set_{i}','domain_1')  
        output_filename = f'universal_sentence_encoder_predictions.parquet'

        test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
        model = get_model_from_pickle(model_directory,model_filename)

        predictions_df,score = predict_with_model(model,test_df)

        score_list.append(score)

        save_parquet(predictions_df,output_directory,output_filename)

        tupla = tuple([i] + score_list)
        listao.append(tupla)

    df_results = pd.DataFrame.from_records(listao)
    df_results.columns = ['text_set','USE']
    
    print(df_results)
    
    EXPERIMENT_RESULT_FOLDER = os.path.join(selected_container,'predictions','experiment_results')
    
    save_parquet(df_results,EXPERIMENT_RESULT_FOLDER,'universal_sentence_encoder_results.parquet')

    logger.info(f'universal sentence encoder results were evaluated')


    return df_results


           
def evaluate_doc_to_vec_predictions(selected_container,text_range,vector_sizes = [32,64,128,256,512]):
    logger.info(f'evaluating doc-2-vec results')
  
    listao = []
    for i in text_range:
        logger.info(f'text {i}')

        score_list = []
        
        for size in vector_sizes:
            logger.info(f'evaluating {size} vectors')

            input_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1')  
            input_filename = f'doc_2_vec_{size}.parquet'

            model_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
            model_filename = f'doc_2_vec_{size}_rf_model.pkl'


            output_directory = os.path.join(selected_container,'predictions','test',f'set_{i}','domain_1')  
            output_filename = f'doc_2_vec_{size}_.parquet'

            test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
            model = get_model_from_pickle(model_directory,model_filename)

            predictions_df,score = predict_with_model(model,test_df)

            score_list.append(score)

            save_parquet(predictions_df,output_directory,output_filename)

        tupla = tuple([i] + score_list)
        listao.append(tupla)

    df_results = pd.DataFrame.from_records(listao)
    df_results.columns = ['text_set','512_dimensoes','256_dimensoes','128_dimensoes','64_dimensoes','32_dimensoes']
        
    EXPERIMENT_RESULT_FOLDER = os.path.join(selected_container,'predictions','experiment_results')
 
    save_parquet(df_results,EXPERIMENT_RESULT_FOLDER,'doc_to_vec.parquet')

    logger.info(f'doc-2-vec results were evaluated')


 
 
        
 

