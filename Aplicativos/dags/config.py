import os
import logging 

replacements = {'the the' : 'the', 
                'to to' : 'to', 
                'thats' : "that\'s", 
                'dont' : "don\'t", 
                'cause' : 'because', 
                'becouse' : 'because', 
                'cuz' : 'because', 
                'bc' : 'because'}



#ESSAY_COLUMNS = ["essay_id","essay_set","essay","domain1_score","domain2_score"]
#SHORT_ANSWER_COLUMNS = ["essay_id","essay_set","essay","Score1","Score2"]

ESSAY_CONTAINER = os.path.join('datalake','essay')
SHORT_ANSWER_CONTAINER = os.path.join('datalake','short_answer')
SHARED_CONTAINER = os.path.join('datalake','shared_container')


ESSAY_TEXT_RANGE = range(1,9)
SHORT_ANSWER_TEXT_RANGE = range(1,11)

#CURRENT_TEXT_RANGE = range(1,9)
CURRENT_TEXT_RANGE = None


RETRAIN_DOC_TO_VEC = False 

LOGLEVEL = logging.INFO


SETTINGS = {'essay':{'container':ESSAY_CONTAINER,'range':list(range(1,9))},
            'short_answer': {'container':SHORT_ANSWER_CONTAINER,'range':list(range(1,11))} 
            }


SETTINGS_FILE = 'settings.json'

LSI_TOPIC_NUMBERS = [10,20,30,40,50,100]


