import pandas as pd
import os 
#from dags import utils 
from dags import config
from dags.utils import *

from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split

import logging 

logger = logging.getLogger(__name__)
logger.setLevel(config.LOGLEVEL)


class text_preprocessor(ABC):

    @abstractmethod
    def get_enriched_texts(self):
        pass

    def __init__(self):
        self.input_file = "corrected_texts.parquet"
        self.output_file = "enriched_texts.parquet"
        self.text_column = "new_text"
        self.score_column = 'score'

    def old_get_train_test(self,df):
        # Shuffle your dataset 
        shuffle_df = df.sample(frac=1)

        # Define a size for your train set 
        train_size = int(0.8 * len(df))

        # Split your dataset 
        df_train = shuffle_df[:train_size].dropna().reset_index(drop = True)
        df_test = shuffle_df[train_size:].dropna().reset_index(drop = True)

        return df_train,df_test

    def get_train_test(self,df):
       
        ## realiza uma divisão de treino e teste separando 8-% para treino e 20% para texte 
        ## como boa prática, fazemos um shuffle para se prevenir de qualquer autocorrelação entre os registros
        ## e também, estatificamos pela variável 'score'
      
        df = self.filter_lower_counts(df,count_to_filter = 10)
        df_train,df_test = train_test_split(df, test_size = 0.2,stratify = df['score'],shuffle = True,)

        ## deleta nulos e reseta o index
        df_train = df_train.dropna().reset_index(drop = True)
        df_test = df_test.dropna().reset_index(drop = True)
 

        return df_train,df_test


    def filter_lower_counts(self,df,count_to_filter):
        ## calcula para cada linha a quantidade de textos daquele conjunto que possuem aquele conceito
        df['quantidade_textos_score_conjunto'] = df.groupby(['text_set',self.score_column])['text_id'].transform('count')
        ## seleciona para trabalharmos somente com os casos em que mais de 10 textos do conjunto obtiveram o conceito
        df = df[df['quantidade_textos_score_conjunto'] >= count_to_filter]

        df = df.drop(columns = ['quantidade_textos_score_conjunto'])

        return df


    def generate_datasets(self):
        df = self.get_enriched_texts()

        df = generate_word_sentence_features(df)
        df.to_parquet(os.path.join(self.output_folder,self.output_file))


        for i in self.text_range:
            domain = 1
            filtro = (df["text_set"] == i) & (df["domain"] == domain)
            df_set = df[filtro]
    
            df_set.drop(columns = "domain")

            df_train, df_test = self.get_train_test(df_set)

            filename_train = f'text_set_{i}_domain_{domain}.parquet'
            save_parquet(df_train,self.train_directory,filename_train)
 
            filename_test = f'text_set_{i}_domain_{domain}.parquet'
            save_parquet(df_test,self.test_directory,filename_test)

        logger.info("All datasets have been generated")


 

class essay_preprocessor(text_preprocessor):

    def __init__(self):
        super().__init__()
        self.input_folder = os.path.join(config.ESSAY_CONTAINER,'raw')
 
        self.output_folder = os.path.join(config.ESSAY_CONTAINER,'raw')

        self.train_directory = os.path.join(config.ESSAY_CONTAINER,'interim','train')
        self.test_directory = os.path.join(config.ESSAY_CONTAINER,'interim','test')

        self.text_range = config.ESSAY_TEXT_RANGE



    def get_enriched_texts(self):
        df = pd.read_parquet(os.path.join(self.input_folder,self.input_file))
        df = df[df[self.text_column] != "-"]
        
        df = df.rename(columns = {"essay_id":"text_id","essay_set":"text_set","essay":"text"})
        df = pd.melt(df,id_vars = ["text_id","text_set","text","new_text"], value_vars = ["domain1_score","domain2_score"]).dropna()
        
        df.columns = ['text_id', 'text_set', 'text', 'new_text', 'domain', 'score']
        
        df["domain"] = df["domain"].replace({"domain1_score":1,"domain2_score":2})
        df = df.dropna()
        
 

        return df




class answer_preprocessor(text_preprocessor):

    def __init__(self):
        super().__init__()
        self.input_folder = os.path.join(config.SHORT_ANSWER_CONTAINER,'raw')
        self.output_folder = os.path.join(config.SHORT_ANSWER_CONTAINER,'raw')

        self.train_directory = os.path.join(config.SHORT_ANSWER_CONTAINER,'interim','train')
        self.test_directory = os.path.join(config.SHORT_ANSWER_CONTAINER,'interim','test')

        self.text_range = config.SHORT_ANSWER_TEXT_RANGE

     
    def get_enriched_texts(self):

        ## lemos o arquivo de short answer e filtramos os textos vazios
        df = pd.read_parquet(os.path.join(self.input_folder,self.input_file))
        df = df[df[self.text_column] != "-"]
        df = df.dropna()

        ## mudamos nomes de colunas pra ter dataset no formato unico para todos os tipos de texto
        df = df.rename(columns = {"Id":"text_id","EssaySet":"text_set","EssayText":"text"})

        ## crio colunas novas, pra colocar no formato unico para todos os tipos de textos
        df["domain"] = 1
        df["score"] = df["Score1"]

        ## seleciono apenas as colunas que existem no formato unico de dataset 
        df = df[['text_id', 'text_set', 'text', 'new_text', 'domain', 'score']]

 

        return df

