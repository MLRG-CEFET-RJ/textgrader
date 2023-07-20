import os, nltk, requests

import pandas as pd

from database_manager import database_manager as db_manager
import configs.configs as configs


def setup_ntlk():
    nltk.download('punkt')

    print('NLTK PUNKT setup finished')


ESSAY_PATH = 'datalake/essay/raw'
SHORT_ANSWER_PATH = 'datalake/short_answer/raw'

ESSAY_FILE = f"{ESSAY_PATH}/essays.xlsx"
SHORT_ANSWER_FILE = f"{SHORT_ANSWER_PATH}/short_answers.xlsx"


def create_datalake_dirs():
    os.makedirs(ESSAY_PATH, exist_ok=True)
    os.makedirs(SHORT_ANSWER_PATH, exist_ok=True)


def download_and_convert_uol_corpus_essays():

    CORPUS_ESSAYS_JSON_LINKS = [
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_01.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_02.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_03.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_04.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_05.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_06.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_07.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_08.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_09.json',
        'https://raw.githubusercontent.com/cassiofb-dev/corpus-redacoes-uol/master/corpus/uoleducacao_redacoes_10.json',
    ]

    CORPUS_ESSAYS_JSONS = [requests.get(corpus_essay_json_link).json() for corpus_essay_json_link in
                           CORPUS_ESSAYS_JSON_LINKS]

    #'essay_id -> id
    # essay_set -: conjunto
    # essay -> redacao
    # rater1_domain1 - > nota /2
    # rater2_domain1 -> nota/2
    # rater3_domain1 -> x
    # domain1_score
    # rater1_domain2
    # rater2_domain2
    # domain2_score
    # rater1_trait1
    # rater1_trait2
    # rater1_trait3
    # rater1_trait4
    # rater1_trait5
    # rater1_trait6
    # rater2_trait1
    # rater2_trait2
    # rater2_trait3
    # rater2_trait4
    # rater2_trait5
    # rater2_trait6
    # rater3_trait1
    # rater3_trait2
    # rater3_trait3
    # rater3_trait4
    # rater3_trait5
    # rater3_trait6'.split(

    corpus_essays_dict = {}

    for column in configs.ESSAY_XLSX_COLUMNS.split(' '):
        corpus_essays_dict[column] = []

    essay_id = 0

    for _, corpus_essays_json in enumerate(CORPUS_ESSAYS_JSONS):
        corpus_essays = corpus_essays_json['redacoes']

        theme = dbManager.create_theme(corpus_essays_json["tema"], corpus_essays_json["data"],
                                       corpus_essays_json["contexto"])

        for essay in corpus_essays:
            essay_id += 1

            if len(essay['texto']) == 0: continue

            corpus_essays_dict['essay_id'].append(essay_id)
            corpus_essays_dict['essay_set'].append(1)
            corpus_essays_dict['essay'].append(essay['texto'])
            corpus_essays_dict['rater1_domain1'].append(essay['nota'] / 2)
            corpus_essays_dict['rater2_domain1'].append(essay['nota'] / 2)
            corpus_essays_dict['domain1_score'].append(essay['nota'])

            dbManager.create_essays(essay, theme, corpus_essays_json["data"])

    # https://stackoverflow.com/questions/61255750/convert-dictionary-of-dictionaries-using-its-key-as-index-in-pandas
    corpus_essay_dataframe = pd.DataFrame.from_dict(
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in corpus_essays_dict.items()])))

    corpus_essay_dataframe.to_excel(ESSAY_FILE, index=False)

if __name__ == '__main__':
    dbManager = db_manager.DatabaseManager()
    dbManager.create_tables()
    setup_ntlk()
    create_datalake_dirs()
    download_and_convert_uol_corpus_essays()
