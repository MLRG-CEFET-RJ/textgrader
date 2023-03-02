 

# Textgrader
### Project elaborated by Ramon Grande Da Luz Bouças as a parcial requirement for obtaining a Bachelor of Science degree in computer science at CEFET/RJ 
### Advisor: Eduardo Bezerra Da Silva D.Sc

# Description
This repository contains TextGrader. In essence textgrader contains the various versions of a Essay and short answer evaluation system.

The system has 4 parts:

1) Preprocessing where we correct spelling change columns schema and do other minor preprocessing steps
2) Feature engineering, where we generate some basic features like word count and sentence count, and generate datasets embedding words 
with each one of the following 4 techniques: TF-IDF, WORD-2-VEC, USE, LSI.
3) Model training, where we train some instances of a random forest model using one of the following 3 approaches: Regression, Classification and Ordinal Classification. 
4) Model Evaluation, where we use the trained models to generate predictions and evaluate those predictions. 

# Requirements 
Our code uses python 3.9 and some libraries like scikit-learn and NLTK. 
We recommend using anaconda or miniconda. If you are using one of these, you can easily setup your environment in order to run our code using the following code

```
conda create -n [choose name] python=3.9
pip install requirements.txt
```

# Datasets

The training datasets used for both essay evaluation task and short answer evaluation task are avaliable at 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7641696.svg)](https://doi.org/10.5281/zenodo.7641696)
The dataset essay.xlsx belongs in the datalake/essay/raw folder 
and the dataset short_answer.xlsx belongs in the datalake/short_answer/raw folder 
However, to run the system there is no need to download and place the datasets at the aforementioned folders, because we already submitted the project with 
the proper files on proper folders. 

 
# Usage

Being in the main directory, do: 

Before running the tasks for evaluating essays or short answers, it is necessary to run the spell corrector for both texts
1) python run.py task general_tasks task_correct_essays
2) python run.py task general_tasks task_correct_short_answers

To run the pipeline for essays run:
python run.py task general_tasks task_pipeline_essays

To run the pipeline for short answer run:
python run.py task general_tasks task_pipeline_short_answer

# Contact

To give your opinion about this work, send an email to ramon.boucas@cefet-rj.br

 
