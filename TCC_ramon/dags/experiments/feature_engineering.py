from msilib.schema import Directory
from dags.utils import * 
 
from dags.feature_engineering.word_embeddings import *
from dags.feature_engineering.latent_semantic_indexing import *
from dags.feature_engineering.tf_idf import * 

import os 



import pandas as pd
from dags.predict.predict import *
from dags import config

import logging 



logger = logging.getLogger(__name__)
logger.setLevel(config.LOGLEVEL)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def generate_lsi_features(selected_container,text_range):
    logging.info("Starting to generate LSI features")
 
    for text_number in text_range:
        logging.info(f'generating features for text {text_number}')

        lsi = LSI_feature_extractor(selected_container,text_number = text_number)
        lsi.generate_all_lsi_topics(text_number = text_number)
        
    logging.info("Generated LSI features")
 


def generate_doc_to_vec_features(selected_container,text_range):
    logging.info("Starting to generate doc-to-vec features")
 
    for text_number in text_range:
        logging.info(f'generating features for text {text_number}')
     
        d2v = doc_2_vec_embedder(selected_container = selected_container,text_number = text_number)
        d2v.load_embedders()
        d2v.generate_all_d2v_embeddings()

    logging.info('Finished to generate embeddings with doc 2 vec')

 
def generate_use_features(selected_container,text_range):
    logging.info("Starting to generate USE features")
 
    for text_number in text_range:
        logging.info(f'generating features for text {text_number}')
     
        use = USE_embedder(selected_container,text_number)
        use.generate_all_use_embeddings()

    logging.info('Generated USE features')



def generate_tf_idf_features(selected_container,text_range):
    tf = TF_IDF(selected_container = selected_container,text_range = text_range)
    tf.generate_tf_idf_features()

 

def generate_features(selected_container,text_range):
    generate_lsi_features(selected_container,text_range)
    generate_doc_to_vec_features(selected_container,text_range)
    generate_use_features(selected_container,text_range)
    generate_tf_idf_features(selected_container,text_range = text_range)

    logging.info('Finished to generate features with all methods')
