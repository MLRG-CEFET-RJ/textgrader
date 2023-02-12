
import os
import pickle
from dags import config
import glob

from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import json
 


def word_count(x):
    lista = word_tokenize(x)
    return len(lista)

def word_count_unique(x):
    lista = set(word_tokenize(x))
    return len(lista)

def sentence_count(x):
    lista = sent_tokenize(x)
    return len(lista)

def generate_word_sentence_features(df,column = 'new_text'):    
    df['word_count'] = df[column].apply(lambda x: word_count(x))
    df['unique_word_count'] = df[column].apply(lambda x: word_count_unique(x))
    df['sentence_count'] = df[column].apply(lambda x: sentence_count(x))
    
    return df


def save_parquet(df,directory,filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
    df.to_parquet(os.path.join(directory,filename))


def save_as_pickle(model,directory,filename):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory,filename)
    
    pickle.dump(model,open(file_path,'wb'))

def get_model_from_pickle(directory,filename):
    
    file_path = os.path.join(directory,filename)

    loaded_model = pickle.load(open(file_path, 'rb'))

    return loaded_model
   

def get_desired_range(df):
    """
    pega o dataframe de textos e dá a quantidade de conjuntos de textos presente, 
    caso o parametro CURRENT_TEXT_RANGE na config esteja habilitado, ele sobrescreve 
    o range desejado
    """
    desired_range = list(df["text_set"].unique())

    if (config.CURRENT_TEXT_RANGE != None):
        desired_range = config.CURRENT_TEXT_RANGE

    return desired_range


def get_files_from_folder(folder):
    file_list = glob.glob(folder + '/*')
    file_list = [i.split("\\")[-1] for i in file_list]
    
    return file_list



def dict_to_json(dictionary,name):
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open(f'{name}', "w") as outfile:
        outfile.write(json_object)


def read_json(filename):
    f = open(filename)

    return f