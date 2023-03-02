from nltk.tokenize import sent_tokenize, word_tokenize
from spellchecker import SpellChecker

import re
from dags import config
import pandas as pd
import os

from abc import ABC,abstractmethod

import logging 

logger = logging.getLogger(__name__)
logger.setLevel(config.LOGLEVEL)

import coloredlogs 

coloredlogs.install()


class spell_corrector(ABC):

    def __init__(self):
        self.output_file =  'corrected_texts.parquet'
        self.new_column = "new_text"

    def replace(self,match):
        return config.replacements[match.group(0)]


    def fix_common_mistakes_in_answer(self,answer):    
        # Example:
        # "This is a text Dr.Henri wrote.This is another text,and it now has one space after the comma."
        # is transformed to
        # 'This is a text Dr. Henri wrote. This is another text, and it now has one space after the comma.'
        answer = re.sub(r'(?<=[.,])(?=[^\s])', r' ', answer)
        
        ## Example:
        ## 'the the cat has this thistle thats long cuz it is.'
        # is transformed to
        ## 'the cat has this thistle that's long because it is.'
        answer = re.sub('|'.join(r'\b%s\b' % re.escape(s) for s in config.replacements), 
                self.replace, answer)

        return answer


    def correct_spell(self,essay):
        ## instancia o objeto spell checker
        spell = SpellChecker()
 
        essay = self.fix_common_mistakes_in_answer(essay)

        ## começa o texto do novo essay
        new_essay = ""

        ## para cada sentença
        for sentence in sent_tokenize(essay):
            ## pega a lista de palavras
            word_list = word_tokenize(sentence)

            ## pega as palavras mal soletradas
            misspelled = spell.unknown(word_list)

            #print(misspelled)

            ## para cada palavra mal soletrada
            for word in misspelled:
                ## obtém a palavra correta
                fixed_word = spell.correction(word)
                #print('%s -> %s' % (word, fixed_word))
                ## corrige a sentença
                sentence = sentence.replace(word, fixed_word)

            new_essay = new_essay + " " + sentence
            
        return new_essay

    def correct_texts(self):
        try:
            df = pd.read_parquet(os.path.join(self.output_directory,self.output_file))
        except:
            df = pd.read_excel(os.path.join(self.input_directory,self.input_file))

            df = df[self.relevant_columns]

            df[self.new_column] = "-"

            df.to_parquet(os.path.join(self.output_directory,self.output_file))
            df = pd.read_parquet(os.path.join(self.output_directory,self.output_file))

        logging.info(f'correcting {self.text_type} texts')
    

        for i in range(len(df)):
            if(df.loc[i,self.new_column] != "-"):
                continue
            df.loc[i,self.new_column] = self.correct_spell(df.loc[i,self.old_column])

            if(i%20 == 0):
                logging.info("corrected text" + str(i))
                df.to_parquet(os.path.join(self.output_directory,self.output_file))
        ## escreve o parquet final
        df.to_parquet(os.path.join(self.output_directory,self.output_file))
    
        logging.info(f'corrected all {self.text_type} texts')




class essay_corrector(spell_corrector):

    def __init__(self):
        super().__init__()

        self.input_directory = os.path.join(config.ESSAY_CONTAINER,'raw')
        self.input_file =  'training_set_rel3.xlsx'
        self.output_directory = os.path.join(config.ESSAY_CONTAINER,'raw')

        self.relevant_columns = ["essay_id","essay_set","essay","domain1_score","domain2_score"]
        self.old_column = "essay"
        self.text_type = "essays"

class short_answer_corrector(spell_corrector):

    def __init__(self):
        super().__init__()

        self.input_directory = os.path.join(config.SHORT_ANSWER_CONTAINER,'raw')
        self.input_file =  'train.xlsx'
        self.output_directory = os.path.join(config.SHORT_ANSWER_CONTAINER,'raw')

        self.relevant_columns = ["Id","EssaySet","EssayText","Score1","Score2"]
        self.old_column = "EssayText"
        self.text_type = "short_answer"
    

 