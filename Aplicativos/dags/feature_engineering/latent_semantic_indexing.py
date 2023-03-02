from csv import list_dialects
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import word_tokenize


import pandas as pd
import os
from dags.utils import * 
from abc import ABC,abstractmethod
from dags import config

import logging 

 
 
class LSI_feature_extractor():

    def __init__(self,base_container):

        BASE_CONTAINER = base_container

        self.column_to_encode = "new_text"
        self.topic_numbers = [10,20,30,40,50,100]

        self.input_train_directory = os.path.join(BASE_CONTAINER,'interim','train')
        self.input_test_directory = os.path.join(BASE_CONTAINER,'interim','test')
      
        self.output_train_directory = os.path.join(BASE_CONTAINER,'processed','train')  
        self.output_test_directory = os.path.join(BASE_CONTAINER,'processed','test') 
 



    def preprocess_data(self,doc_set):
        texts = []
        for essay in doc_set:
            ## tokeniza o ensaio
            tokenized = word_tokenize(essay)
            ## adiciona a lista de palavras na lista de listas que representa o corpus
            texts.append(tokenized)
        return texts


    def generate_topic_object(self,texts,num_topics):
        ## constroi o dicionario a partir da lista de listas que representa o nosso corpus
        dictionary = corpora.Dictionary(texts)

        ## constroi a matriz termo-documento a partir da lista de listas que representa o nosso corpus
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in texts]

        ## constroi o modelo LSI a partir da matriz termo-documento, do numero de tópicos e do dicionario
        lsamodel = LsiModel(doc_term_matrix,num_topics = num_topics, id2word = dictionary)

        ## usa o modelo LSI para obter o objeto mapeando os documentos para os topicos
        doc_to_topic_list = lsamodel[doc_term_matrix]

        return doc_to_topic_list


    def gensim_object_to_dataframe(self,doc_to_topic_list):
        ## pega uma lista de tuplas chave-valor como [(1,10),(2,100),(3,1000)] e transforma em uma tupla com os valores 
        ## como por exemplo (10,100,1000)
        func = lambda x:tuple(v for k,v in x)

        ## para cada documento , ou seja, lista de tuplas (feature,valor), transforma em tupla unica com todos os valores
        ## e monta um listao de duplas
        list_of_tuples = [func(document) for document in doc_to_topic_list]

        df = pd.DataFrame.from_records(list_of_tuples)
        
        return df

    def generate_topics(self,texts,num_topics):
        doc_to_topic_object = self.generate_topic_object(texts,num_topics = num_topics)
        topic_dataframe = self.gensim_object_to_dataframe(doc_to_topic_object)
        
        return topic_dataframe


    def generate_topic_features(self,df,num_topics):
        lists = self.preprocess_data(df[self.column_to_encode])
        df_topics = self.generate_topics(lists,num_topics = num_topics)
        df = pd.concat([df,df_topics],axis = 1)

        ## converte os nomes das colunas para string 
        df.columns = df.columns.astype(str)
        
        return df


def generate_topics_datasets(selected_container,text_range):
    logging.info("Starting to generate LSI features")

    lsi = LSI_feature_extractor(selected_container)

    for topic_number in config.LSI_TOPIC_NUMBERS:
        logging.info(f'generating LSI features for {topic_number} topics')

        for i in text_range:
            logging.info(f'text {i}')


            input_filename = f'text_set_{i}_domain_1.parquet'
        
            df_train = pd.read_parquet(os.path.join(lsi.input_train_directory,input_filename))
            df_test =  pd.read_parquet(os.path.join(lsi.input_test_directory,input_filename))

            logging.debug(f'{len(df_train)} examples on train dataset')
            logging.debug(f'{len(df_train)} examples on test dataset')

            embedded_train = lsi.generate_topic_features(df_train,num_topics = topic_number)
            embedded_test =  lsi.generate_topic_features(df_test,num_topics = topic_number)

            output_filename = f'lsi_{topic_number}_topics.parquet'

            output_train_path = os.path.join(lsi.output_train_directory,f'set_{i}',f'domain_1')
            output_test_path = os.path.join(lsi.output_test_directory,f'set_{i}',f'domain_1')

            save_parquet(embedded_train,output_train_path,output_filename)
            save_parquet(embedded_test,output_test_path,output_filename)
        logging.info("Generated LSI features")






