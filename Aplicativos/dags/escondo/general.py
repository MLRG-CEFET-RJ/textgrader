
from dags.doc_to_vec_trainer import *
from dags.experiments import *
from dags.word_embeddings import *
from dags.latent_semantic_indexing import *
from dags.spell_correct import *
from dags.general_preprocessing import *
import logging
logger = logging.getLogger(__name__)

## tasks específicas de essays

def run_dag(task):
    from dags.runner import run_task

    run_task(task)

def task_correct_essays():
    ec = essay_corrector()
    ec.correct_texts()
    
def task_generate_essay_datasets():
    ep = essay_preprocessor()
    ep.generate_datasets()


## tasks especificas de short answer

def task_correct_short_answers():
    sac = short_answer_corrector()
    sac.correct_texts()

def task_generate_short_answer_datasets():
    sap = answer_preprocessor()
    sap.generate_datasets()
    

## tasks gerais 

    
def task_generate_features(selected_container):
    generate_d2v_embeddings(selected_container)
    generate_use_embeddings(selected_container)
    
    lsi_fe = LSI_feature_extractor(selected_container)
    lsi_fe.generate_topics_datasets()


def task_train_models(selected_container):
    experiments_with_lsi_topics(selected_container)
    experiments_with_doc_2_vec(selected_container)
    experiments_with_universal_sentence_encoder(selected_container)


def task_evaluate_models(selected_container):
    evaluate_lsi_predictions(selected_container)
    evaluate_universal_sentence_encoder_predictions(selected_container)
    evaluate_doc_to_vec_predictions(selected_container)


def task_train_doc_2_vec():
    train_doc_to_vec()

def task_shared_tasks(selected_container):
    if config.RETRAIN_DOC_TO_VEC:
        task_train_doc_2_vec()
        print("word 2 vec treinado")

    task_generate_features(selected_container)
    print("gerou as features de todos os métodos")
    task_train_models(selected_container)
    print("os modelos foram treinados")
    task_evaluate_models(selected_container)
    print("a avaliação dos modelos foi feita")
    


def task_pipeline_essays():
    task_correct_essays()
    logger.info("corrigiu os essays")
    task_generate_essay_datasets()
    task_shared_tasks(selected_container = config.ESSAY_CONTAINER)


def task_pipeline_short_answer():
    task_correct_short_answers()
    logger.info("corrigiu as respostas curtas")

    task_generate_short_answer_datasets()
    logger.info("gerou os datasets de resposta curta")

    task_shared_tasks(selected_container = config.ESSAY_CONTAINER)
    logger.info("gerou os datasets de resposta curta")





