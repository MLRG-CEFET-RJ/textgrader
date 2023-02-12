
from cgitb import text
import logging
from dags.preprocessing.spell_correct import *
from dags.preprocessing.general_preprocessing import *

from dags.experiments.feature_engineering import generate_features


from dags.model_training.doc_to_vec_trainer import *
from dags.experiments.modeling import *
from dags.experiments.predicting import *


from dags import config 
from dags import utils




logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

## tasks específicas de essays


def run_dag(task):
    from dags.runner import run_task

    run_task(task)



def task_correct_essays():
    ec = essay_corrector()
    ec.correct_texts()
    logger.info("finished to correct essays")
    
def task_generate_essay_datasets():
    ep = essay_preprocessor()
    ep.generate_datasets()
    logger.info("finished to generate essay datasets")



## tasks especificas de short answer

def task_correct_short_answers():
    sac = short_answer_corrector()
    sac.correct_texts()

def task_generate_short_answer_datasets():
    sap = answer_preprocessor()
    sap.generate_datasets()
    

## tasks gerais 

    
def task_generate_features(selected_container,text_range):
    generate_features(selected_container,text_range)

def task_train_regression_models(selected_container,text_range):

    rmt = regression_model_trainer(selected_container,text_range)

    logger.info('starting to train regression models')
   
    rmt.experiments()
 
 
    logger.info('finished training regression models')

def task_evaluate_regression_models(selected_container,text_range):

    rmt = regression_model_trainer(selected_container,text_range)

    logger.info('evaluating regression models')
   
   
    rmt.evaluate()
 
    logger.info('finished evaluating regression models')

def task_evaluate_classification_models(selected_container,text_range):

    cmt = classification_model_trainer(selected_container,text_range)

    logger.info('evaluating classification models')
   
   
    cmt.evaluate()
 
 
    logger.info('finished evaluating classification models')

def task_evaluate_ordinal_classifier_models(selected_container,text_range):

    ocmt = ordinal_classification_model_trainer(selected_container,text_range)

    logger.info('evaluating ordinal classification models')
   
   
    ocmt.evaluate()
 
 
    logger.info('finished evaluating ordinal classification models')
  
  
def task_train_classifier_models(selected_container,text_range):

    rmt = classification_model_trainer(selected_container,text_range)

    logger.info('starting to train classification models')

    rmt.experiments()
  
    logger.info('finished training classification models')

def task_train_ordinal_classifier_models(selected_container,text_range):

    ocmt = ordinal_classification_model_trainer(selected_container,text_range)

    logger.info('starting to train ordinal classification models')

    ocmt.experiments()
  
    logger.info('finished training ordinal classification models')


def task_train_models(selected_container,text_range):
    task_train_regression_models(selected_container,text_range)
    task_train_classifier_models(selected_container,text_range)
    task_train_ordinal_classifier_models(selected_container,text_range)



def task_evaluate_models(selected_container, text_range):
    task_evaluate_regression_models(selected_container, text_range)
    task_evaluate_classification_models(selected_container, text_range)
    task_evaluate_ordinal_classifier_models(selected_container,text_range)

def task_train_doc_2_vec():
    train_doc_to_vec()

def task_shared_tasks(selected_container,text_range):
    if config.RETRAIN_DOC_TO_VEC:
        task_train_doc_2_vec()
        print("word 2 vec treinado")

    task_generate_features(selected_container,text_range = text_range)
    task_train_models(selected_container,text_range= text_range)
    task_evaluate_models(selected_container, text_range)
 


def save_settings(text_mode):
    settings = config.SETTINGS
    especial_settings = settings[text_mode]

    dict_to_json(especial_settings,config.SETTINGS_FILE)

    
def task_pipeline_essays():
    task_correct_essays()
    task_generate_essay_datasets()
    task_shared_tasks(selected_container = config.ESSAY_CONTAINER,text_range = config.ESSAY_TEXT_RANGE)


def task_pipeline_short_answer():
    task_correct_short_answers()
    logger.info("corrigiu as respostas curtas")

    task_generate_short_answer_datasets()
    print("gerou os datasets de resposta curta")

    task_shared_tasks(selected_container = config.SHORT_ANSWER_CONTAINER,text_range = config.SHORT_ANSWER_TEXT_RANGE)
    print("gerou os datasets de resposta curta")



def task_pipeline_correct_all():
    task_correct_essays()
    task_correct_short_answers()





