from msilib.schema import Directory
from dags.utils import * 
from dags.model_training.train_model import pipeline_random_forest
from dags.feature_engineering.word_embeddings import *
from dags.feature_engineering.latent_semantic_indexing import *

import os 



import pandas as pd
from dags.predict.predict import *
from dags import config

import logging 



logger = logging.getLogger(__name__)
logger.setLevel(config.LOGLEVEL)

 
def generate_topics_datasets(selected_container,text_range):
    logger.info("Starting to generate LSI features")

    lsi = LSI_feature_extractor(selected_container)

    for topic_number in config.LSI_TOPIC_NUMBERS:
        logger.info(f'generating LSI features for {topic_number} topics')

        for i in text_range:
            logger.info(f'text {i}')
            logger.info(f'text_{i}')


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


