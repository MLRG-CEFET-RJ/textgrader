from msilib.schema import Directory
from dags.utils import * 
from dags.model_training.train_model import pipeline_random_forest
from dags.feature_engineering.word_embeddings import *
from dags.feature_engineering.latent_semantic_indexing import *

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

def experiments_with_lsi_topics(selected_container,text_range):
    logger.info("starting to train models with LSI features")
    for topic_number in [10,20,30,40,50,100]:
        logger.info(f"training models with the {topic_number} dataset")

        print(topic_number)

    
        for i in text_range:
            input_directory = os.path.join(selected_container,'processed','train',f'set_{i}','domain_1')  
            input_filename = f'lsi_{topic_number}_topics.parquet'
            
            print(i)

            logger.info(f"text {i}")
           
            df = pd.read_parquet(os.path.join(input_directory,input_filename))
            df = df.dropna()
            
            model = pipeline_random_forest(df)
            
            output_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
            output_filename = f'lsi_{topic_number}_topics_rf_model.pkl'
            
            save_as_pickle(model,output_directory,output_filename)
        logger.info("finished training models with LSI features")

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


def experiments_with_universal_sentence_encoder(selected_container,text_range):
    logger.info("starting to train models with universal sentence encoder features")
    for i in text_range:
        input_directory = os.path.join(selected_container,'processed','train',f'set_{i}','domain_1')  
        input_filename = 'universal_sentence_encoder.parquet'

        df = pd.read_parquet(os.path.join(input_directory,input_filename))
        df = df.dropna()

        model = pipeline_random_forest(df)

        output_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
        output_filename = 'universal_sentence_encoder_rf_model.pkl'

        save_as_pickle(model,output_directory,output_filename)
    logger.info("finished training models with universal sentence encoder features")


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


def experiments_with_doc_2_vec(selected_container,text_range):
    logger.info("starting to train models with doc-2-vec features")
    for size in [512,256,128,64,32]:
        logger.info(f'training model with {size} doc-2-vec features')
        for i in text_range:
            input_directory = os.path.join(selected_container,'processed','train',f'set_{i}','domain_1')  
            input_filename = f'doc_2_vec_{size}.parquet'

            logger.info(f'text {i}')

            df = pd.read_parquet(os.path.join(input_directory,input_filename))
            df = df.dropna()

            model = pipeline_random_forest(df)

            output_directory = os.path.join(selected_container,'model',f'set_{i}','domain_1')  
            output_filename = f'doc_2_vec_{size}_rf_model.pkl'

            save_as_pickle(model,output_directory,output_filename)
    logger.info("finished training model with doc 2 vec features")

            
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


def generate_topics_datasets(selected_container,text_range):
    logger.info("Starting to generate LSI features")

    lsi = LSI_feature_extractor(selected_container)

    for topic_number in config.LSI_TOPIC_NUMBERS:
        logger.info(f'generating LSI features for {topic_number} topics')

        for i in text_range:
            logger.info(f'text {i}')
             
            input_filename = f'text_set_{i}_domain_1.parquet'
        
            df_train = pd.read_parquet(os.path.join(lsi.input_train_directory,input_filename))
            df_test =  pd.read_parquet(os.path.join(lsi.input_test_directory,input_filename))

            logger.debug(f'{len(df_train)} examples on train dataset')
            logger.debug(f'{len(df_train)} examples on test dataset')

            embedded_train = lsi.generate_topic_features(df_train,num_topics = topic_number)
            embedded_test =  lsi.generate_topic_features(df_test,num_topics = topic_number)

            output_filename = f'lsi_{topic_number}_topics.parquet'

            output_train_path = os.path.join(lsi.output_train_directory,f'set_{i}',f'domain_1')
            output_test_path = os.path.join(lsi.output_test_directory,f'set_{i}',f'domain_1')

            save_parquet(embedded_train,output_train_path,output_filename)
            save_parquet(embedded_test,output_test_path,output_filename)
        logger.info("Generated LSI features")


def generate_use_embeddings(selected_container,text_range):
    use = USE_embedder()

    logger.info('Starting to generate embeddings with universal sentence encoder')

    print("deveria ter logado")

    for i in text_range:       
        logger.info(f'text {i}')

        input_train_directory = os.path.join(selected_container,'interim','train')
        input_test_directory = os.path.join(selected_container,'interim','test')

        input_filename = f'text_set_{i}_domain_1.parquet'

        df_train = pd.read_parquet(os.path.join(input_train_directory,input_filename))
        df_test =  pd.read_parquet(os.path.join(input_test_directory,input_filename))

        embedded_train = use.generate_embed_features(df_train)
        embedded_test =  use.generate_embed_features(df_test)

        output_train_directory = os.path.join(selected_container,'processed','train',f'set_{i}','domain_1')  
        output_test_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1') 
        output_filename = f'universal_sentence_encoder.parquet'

        save_parquet(embedded_train,output_train_directory,output_filename)
        save_parquet(embedded_test,output_test_directory,output_filename)
    
    logger.info('generated embeddings with universal sentence encoder')



def generate_d2v_embeddings(selected_container,text_range,vector_list = [512,256,128,64,32]):
    for vector_size in vector_list:
        logger.info(f'generating embeddings with size {vector_size} with doc 2 vec')
        d2v = doc_2_vec_embedder(vector_size)

        for i in text_range:
            
            logger.info(f'text {i}')
    
            input_train_directory = os.path.join(selected_container,'interim','train')
            input_test_directory = os.path.join(selected_container,'interim','test')

            input_filename = f'text_set_{i}_domain_1.parquet'

            df_train = pd.read_parquet(os.path.join(input_train_directory,input_filename))
            df_test =  pd.read_parquet(os.path.join(input_test_directory,input_filename))


            embedded_train = d2v.generate_embed_features(df_train)
            embedded_test =  d2v.generate_embed_features(df_test)

            output_train_directory = os.path.join(selected_container,'processed','train',f'set_{i}','domain_1')  
            output_test_directory = os.path.join(selected_container,'processed','test',f'set_{i}','domain_1') 
            output_filename = f'doc_2_vec_{vector_size}.parquet'

            save_parquet(embedded_train,output_train_directory,output_filename)
            save_parquet(embedded_test,output_test_directory,output_filename)

    logger.info('Finished to generate embeddings with doc 2 vec')




    
def generate_features(selected_container,text_range):
    generate_d2v_embeddings(selected_container,text_range = text_range)
    generate_use_embeddings(selected_container,text_range = text_range)
 
    generate_topics_datasets(selected_container,text_range = text_range)

    logger.info('Finished to generate features with all methods')


