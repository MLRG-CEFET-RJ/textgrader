
import gensim.downloader as api
import pandas as pd
from nltk.tokenize import word_tokenize
from dags.utils import *
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from dags import config

def save_essay_corpus():
    df_essay = pd.read_parquet(os.path.join(config.ESSAY_CONTAINER,'raw','corrected_texts.parquet'))
    lista_ensaios = list(df_essay[df_essay["new_text"] != "-"]["new_text"])
    essay_corpora = [word_tokenize(i) for i in lista_ensaios]
    save_as_pickle(essay_corpora,os.path.join(config.SHARED_CONTAINER,'model','word_embedding','corpora'),'essay_corpora.pkl')
    
    return essay_corpora


def save_short_answer_corpus():
    df_answers = pd.read_parquet(os.path.join(config.SHORT_ANSWER_CONTAINER,'raw','corrected_texts.parquet'))
    lista_ensaios = list(df_answers[df_answers["new_text"] != "-"]["new_text"])
    short_answer_corpora = [word_tokenize(i) for i in lista_ensaios]
    save_as_pickle(short_answer_corpora,os.path.join(config.SHARED_CONTAINER,'model','word_embedding','corpora'),'short_answer_corpora.pkl')


def get_tagged_document_corpora():
    essay_corpus = get_model_from_pickle(os.path.join(config.SHARED_CONTAINER,'model','word_embedding','corpora'),'essay_corpora.pkl')
    short_answer_corpus = get_model_from_pickle(os.path.join(config.SHARED_CONTAINER,'model','word_embedding','corpora'),'short_answer_corpora.pkl')
    
    general_corpus = essay_corpus + short_answer_corpus
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(general_corpus)]
    
    return documents


def generate_word_to_vec_models(documents):
    for size in [512,256,128,64,32]:
        ## uso a versão PV-DBOW pois é mencionada no artigo que é a versão mais rápida
        model = Doc2Vec(documents, min_count=1, dm = 0, vector_size = size)
        save_as_pickle(model,os.path.join(config.SHARED_CONTAINER,'model','word_embedding'),f'doc_2_vec_{size}.pkl')


def train_doc_to_vec():
    save_essay_corpus()
    save_short_answer_corpus()
    documents = get_tagged_document_corpora()
    generate_word_to_vec_models(documents)

