from msilib.schema import Directory
import select
import pandas as pd
import tensorflow_hub as hub
import os
from dags.utils import *
import numpy as np
from nltk.tokenize import word_tokenize
from dags import config
import logging

 
import logging
 

# create logger
logger = logging.getLogger(__name__)


import tensorflow as tf
tf.get_logger().setLevel('ERROR')
 
 
class text_embedder():


    def generate_embed_features(self,df,coluna = "new_text"):
        df_embed = self.embed_words(df[coluna])
        df = pd.concat([df,df_embed], axis = 1)
        df = df.drop(columns = [coluna], errors = "ignore")
        df.columns = df.columns.astype(str)

        return df

class USE_embedder(text_embedder):
    def __init__(self,selected_container,text_number):
        super().__init__()
        self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.selected_container = selected_container
        self.text_number = text_number
        self.input_filename = f'text_set_{self.text_number}_domain_1.parquet'

       

    def embed_words(self,word_vector):
        embeddings = self.embedder(word_vector)
        embed_features = pd.DataFrame(embeddings.numpy())

        return embed_features

    def generate_all_use_embeddings(self):
        input_train_directory = os.path.join(self.selected_container,'interim','train')
        input_test_directory = os.path.join(self.selected_container,'interim','test')

        input_filename = f'text_set_{self.text_number}_domain_1.parquet'

        df_train = pd.read_parquet(os.path.join(input_train_directory,input_filename))
        df_test =  pd.read_parquet(os.path.join(input_test_directory,input_filename))

        embedded_train = self.generate_embed_features(df_train)
        embedded_test =  self.generate_embed_features(df_test)

        output_train_directory = os.path.join(self.selected_container,'processed','train',f'set_{self.text_number}','domain_1','USE','version_unique')  
        output_test_directory = os.path.join(self.selected_container,'processed','test',f'set_{self.text_number}','domain_1','USE','version_unique') 
        output_filename = f'features.parquet'

        save_parquet(embedded_train,output_train_directory,output_filename)
        save_parquet(embedded_test,output_test_directory,output_filename)

    
class doc_2_vec_embedder(text_embedder):
    def __init__(self,selected_container,text_number):
        super().__init__()
        self.model_directory = os.path.join(config.SHARED_CONTAINER,'model','word_embedding')
        self.vector_list = [512,256,128,64,32]
        self.selected_container = selected_container
        self.text_number = text_number
        self.input_filename = f'text_set_{self.text_number}_domain_1.parquet'

    def get_embedder(self,vector_size):
        """
        obtem o embedder pré-treinado
        """

        filename = f'doc_2_vec_{vector_size}.pkl'
        embedder = get_model_from_pickle(self.model_directory,filename)

        return embedder

    def load_embedders(self):
        """
        Carrega os embedders pré-treinados em um atributo da classe
        """
        embedders = {}

        for i in self.vector_list:
            embedders[i] = self.get_embedder(i)

        self.embedders = embedders

    
    def embed_one_text(self,text,embedder):
        """
        utiliza o embedder para obter a representação vetorial de um texto
        """
        words = word_tokenize(text)
        vector = embedder.infer_vector(words)
        
        return vector
    
    def embed_words(self,word_vector,embedder):
        """
        obtem a representação vetorial de todos os textos a partir da coluna de texto
        """
        serie = word_vector.apply(lambda x:self.embed_one_text(x,embedder= embedder))
        df = pd.DataFrame.from_records(np.array(serie))
    
        return df


    def generate_embed_features(self,df,embedder,coluna = "new_text"):
        """
        junta as features de representação vetorial as outras features do dataframe
        """

        df_embed = self.embed_words(df[coluna],embedder = embedder)
        df = pd.concat([df,df_embed], axis = 1)
        df = df.drop(columns = [coluna], errors = "ignore")
        df.columns = df.columns.astype(str)

        return df


    def generate_all_d2v_embeddings(self):
        """
        executa a representação vetorial para todas as dimensionalidades
        """

        for vector_size in self.vector_list:

            input_train_directory = os.path.join(self.selected_container,'interim','train')
            input_test_directory = os.path.join(self.selected_container,'interim','test')


            df_train = pd.read_parquet(os.path.join(input_train_directory,self.input_filename))
            df_test =  pd.read_parquet(os.path.join(input_test_directory,self.input_filename))

            
            output_train_directory = os.path.join(self.selected_container,'processed','train',f'set_{self.text_number}','domain_1','doc_2_vec',f'version_{vector_size}')  
            output_test_directory = os.path.join(self.selected_container,'processed','test',f'set_{self.text_number}','domain_1','doc_2_vec',f'version_{vector_size}')  
          
                
            logging.info(f'generating features with size {vector_size}')

            embedder = self.embedders[vector_size]

            embedded_train = self.generate_embed_features(df_train,embedder = embedder)
            embedded_test =  self.generate_embed_features(df_test,embedder = embedder)

            output_filename = f'features.parquet'

            save_parquet(embedded_train,output_train_directory,output_filename)
            save_parquet(embedded_test,output_test_directory,output_filename)

            


    







