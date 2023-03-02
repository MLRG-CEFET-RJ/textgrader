import numpy as np
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from dags.utils import *


def predict_with_model(model,test_df,predicted_variable = "score"):
    X = test_df[model.feature_names_in_]


    y_pred = model.predict(X)
    y_pred = np.round(y_pred)

    test_df["prediction"] = y_pred

    score = cohen_kappa_score(test_df["prediction"],test_df[predicted_variable], weights = "quadratic")
    
    return test_df , score


def evaluate_lsi_predictions(topic_numbers = [10,20,30,40,50,100]):

    experiments_folder = os.path.join('essay','results')

    listao = []

    for i in range(1,9):
        score_list = []
        for topic in topic_numbers:
            input_directory = os.path.join('essay','processed','test',f'set_{i}','domain_1')  
            input_filename = f'lsi_{topic}_topics.parquet'

            model_directory = os.path.join('essay','model',f'set_{i}','domain_1')  
            model_filename = f'lsi_{topic}_topics_rf_model.pkl'

            print(i)
            print(model_filename)

            output_directory = os.path.join('essay','predictions','test',f'set_{i}','domain_1')  
            output_filename = f'lsi_{topic}_topics_predictions.parquet'

            test_df = pd.read_parquet(os.path.join(input_directory,input_filename)).dropna()
            model = get_model_from_pickle(model_directory,model_filename)

            predictions_df,score = predict_with_model(model,test_df)

            score_list.append(score)

            save_parquet(predictions_df,output_directory,output_filename)

        tupla = tuple([i] + score_list)
        listao.append(tupla)

        df_results = pd.DataFrame.from_records(listao)
        df_results.columns = ['essay_set'] + [f'{topic}_topicos' for topic in topic_numbers]

        df_results.to_parquet(os.path.join(experiments_folder,'lsi_topics_results.parquet'))
        
    return df_results














