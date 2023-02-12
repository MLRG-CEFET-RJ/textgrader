from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import pandas as pd
import pickle
 
import os
from dags import config
import pandas as pd
from dags.utils import *
import logging
from dags import config

class TF_IDF():
    def __init__(self,selected_container,text_range):
        self.selected_container = selected_container
 
        self.vectorizer_folder = os.path.join(self.selected_container,'model', 'tf_idf_vectorizer')
        ## salva o vetorizador treinado
        self.file_path = os.path.join(self.vectorizer_folder,'tf_idf_vectorizer.pkl')
        self.text_range = text_range


    def fit_vectorizer(self,df_train,text_number):
        """
        Método responsável por treinar o vetorizador dos textos
        """
        ## obtém a lista de textos do conjunto usado para treinar o vetorizador
        lista = list(df_train['new_text'])

        ## cria o vetorizador que utiliza TF-IDF
        tv = TfidfVectorizer(stop_words = 'english',min_df = 10)

        ## treina o vetorizador com o conjunto fornecido para treino
        vectorizer = tv.fit(lista)

        folder = os.path.join(self.vectorizer_folder,f'set_{text_number}') 
        save_as_pickle(vectorizer,folder,'vectorizer.pkl')
        
        return vectorizer
    
      

    def vectorize_data(self,df,text_number):
        """
        Carrega o vetorizador treinado e utiliza o mesmo, com o intuito de realizar a vetorização dos dados
        """
        ## carrega o modelo treinado que queremos usar para realizar a vetorização

        file_path = os.path.join(self.vectorizer_folder,f'set_{text_number}','vectorizer.pkl') 

        loaded_model = pickle.load(open(file_path, 'rb'))


        ## obtém a lista de textos do conjunto em que queremos aplicar o vetorizador
        lista = list(df['new_text'])
        
        ## realiza efetivamente a vetorização, transformando em uma matriz esparsa
        X = loaded_model.transform(lista)

        ## transforma a matriz esparsa em um dataframe organizado com as frequencias TF-IDF das palavras 
        df_vetorizado = pd.DataFrame(X.A, columns=loaded_model.get_feature_names_out())
        
        ## remove do datafame vetorizado palavras do texto que tenham coincidencia de nome com features (como text,essay, e outras)
        ## isso é imprtante, pois ao concatenarmos depois, não haver colunas duplicadas
        df_vetorizado = df_vetorizado.drop(columns = list(df.columns),errors = 'ignore')
        
        return df_vetorizado

    def get_selected_features(self,df,n = 512):
        """
        obtém o dataframe já vetorizado com TF-IDF e emprega um método de seleção de features
        baseado em features com as maiores variâncias para selecionar as features
        """

        ## cria um dataframe com o desvio padrão das features
        words_df = pd.DataFrame(df.std()).reset_index()
        ## renomeia as colunas
        words_df.columns = ['word','std']
        ## ordena as colunas de forma decrescente pelo desvio padrão 
        words_df = words_df.sort_values(by = 'std', ascending = False)
        ## seleciona as n colunas com maior desvio padrão 
        words_df = words_df.iloc[0:n,:]

        ## retorna a lista das palavras mais significantes
        words_list = list(words_df['word'].unique())
        
        return words_list

    def filter_dataframe(self,df_train,df_test, n_features):
        words_list = self.get_selected_features(df_train,n = n_features)

        filtered_train = df_train[words_list]
        filtered_test = df_test[words_list]
        
        return filtered_train, filtered_test

    
    def generate_tf_idf_features(self):
    
        train_input_folder = os.path.join(self.selected_container,'interim','train')
        test_input_folder = os.path.join(self.selected_container,'interim','test')

        train_output_folder = os.path.join(self.selected_container,'processed','train')
        test_output_folder = os.path.join(self.selected_container,'processed','test')

        
        for i in self.text_range:
            logging.info(f'generating features with tf-idf for text {i}')
            df_train = pd.read_parquet(os.path.join(train_input_folder,f'text_set_{i}_domain_1.parquet'))
            df_test = pd.read_parquet(os.path.join(test_input_folder,f'text_set_{i}_domain_1.parquet'))


            ## fito o vetorizador com base exclusivamente no conjunto de treino
            vectorizer = self.fit_vectorizer(df_train,text_number = i)
            
            ## gero a versão vetorizada do conjunto de treino 
            vectorized_train = self.vectorize_data(df_train,text_number = i)
            output_train_directory = os.path.join(train_output_folder,f'set_{i}','domain_1','TF_IDF')
            #save_parquet(vectorized_train,output_train_directory,'tf_idf_all_features.parquet')

            ## gera a versão vetorizada do conjunto de teste
            vectorized_test = self.vectorize_data(df_test,text_number = i)
            output_test_directory = os.path.join(test_output_folder,f'set_{i}','domain_1','TF_IDF')
            #save_parquet(vectorized_test,output_test_directory,os.path.join('ALL',tf_idf_all_features.parquet')


            for n_features in [512,256,128,64,32]:
                logging.info(f'generating tf-idf with {n_features} features')
                filtered_train,filtered_test = self.filter_dataframe(vectorized_train,vectorized_test,n_features = n_features)

                df_train = df_train.drop(columns = ['new_text'], errors = 'ignore')
                filtered_train = pd.concat([df_train,filtered_train], axis = 1)

                df_test = df_test.drop(columns = ['new_text'], errors = 'ignore')
                filtered_test = pd.concat([df_test,filtered_test], axis = 1)

                train_path = os.path.join(output_train_directory,f'version_{n_features}')
                test_path = os.path.join(output_test_directory,f'version_{n_features}')


                save_parquet(filtered_train,train_path,'features.parquet')
                save_parquet(filtered_train,test_path,'features.parquet')

        
    