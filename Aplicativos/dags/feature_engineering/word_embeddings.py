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
 
 
class text_embedder():

    def generate_embed_features(self,df,coluna = "new_text"):
        df_embed = self.embed_words(df[coluna])
        df = pd.concat([df,df_embed], axis = 1)
        df = df.drop(columns = [coluna], errors = "ignore")
        df.columns = df.columns.astype(str)

        return df

class USE_embedder(text_embedder):
    def __init__(self):
        super().__init__()
        self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def embed_words(self,word_vector):
        embeddings = self.embedder(word_vector)
        embed_features = pd.DataFrame(embeddings.numpy())

        return embed_features

    
class doc_2_vec_embedder(text_embedder):
    def __init__(self,vector_size):
        super().__init__()
        directory = os.path.join(config.SHARED_CONTAINER,'model','word_embedding')
        filename = f'doc_2_vec_{vector_size}.pkl'
        self.embedder = get_model_from_pickle(directory,filename)


    def embed_one_text(self,text):
        words = word_tokenize(text)
        vector = self.embedder.infer_vector(words)
        
        return vector


    def embed_words(self,word_vector):
        serie = word_vector.apply(lambda x:self.embed_one_text(x))
        df = pd.DataFrame.from_records(np.array(serie))
    
        return df

 







