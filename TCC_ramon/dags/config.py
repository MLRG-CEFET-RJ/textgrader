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


RETRAIN_DOC_TO_VEC = True 

LOGLEVEL = logging.INFO


SETTINGS = {'essay':{'container':ESSAY_CONTAINER,'range':list(range(1,9))},
            'short_answer': {'container':SHORT_ANSWER_CONTAINER,'range':list(range(1,11))} 
            }


SETTINGS_FILE = 'settings.json'

LSI_TOPIC_NUMBERS = [10,20,30,40,50,100]


 


SHORT_ANSWER_SCORE_RANGE = {1:{'min_score':0,'max_score':3},
2:{'min_score':0,'max_score':3},
3:{'min_score':0,'max_score':2},
4:{'min_score':0,'max_score':2},
5:{'min_score':0,'max_score':3},
6:{'min_score':0,'max_score':3},
7:{'min_score':0,'max_score':2},
8:{'min_score':0,'max_score':2},
9:{'min_score':0,'max_score':2},
10:{'min_score':0,'max_score':2},
}



ESSAY_SCORE_RANGE = {1:{'min_score':2,'max_score':12},
2:{'min_score':1,'max_score':6},
3:{'min_score':0,'max_score':3},
4:{'min_score':0,'max_score':3},
5:{'min_score':0,'max_score':4},
6:{'min_score':0,'max_score':4},
7:{'min_score':0,'max_score':30},
8:{'min_score':0,'max_score':60},
 
}




