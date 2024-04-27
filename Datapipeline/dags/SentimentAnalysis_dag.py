from airflow import DAG,Dataset
from datetime import datetime
from airflow.operators.python import PythonOperator
import numpy as np
import pandas as pd
from collections import Counter
#from nltk.corpus import stopwords
from ML.data_analysis import data_analysis
from ML.data_preprocessing import data_preprocessing
from ML.training import data_training
import os
from airflow.models import Variable


TRAIN_FILE = '/opt/airflow/dataset/train.csv'
TEST_FILE = '/opt/airflow/dataset/test.csv'
ANALYSIS_TRAIN_FILE = '/opt/airflow/dataset/analysis_train.csv'
PREPROCESSED_FILE='/opt/airflow/dataset/cleaned_train.csv'
#TRAINED_FILE='/opt/airflow/dataset/trained_data.csv'

def files_exists():
    train_path = Variable.get("train_file_path", TRAIN_FILE)
    test_path = Variable.get("test_file_path", TEST_FILE)
    
    # Check if the train file exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"The specified train file does not exist: {train_path}")
    
    # Check if the test file exists
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"The specified test file does not exist: {test_path}")
    
    #if train_path and test_path:
    return True


# def load_data():
#     train=Dataset("/opt/airflow/train.csv")
#     test=Dataset("/opt/airflow/test.csv")
#     return True

# def data_analysis():
#     data_analysis("/opt/airflow/dataset/train.csv","/opt/airflow/dataset/test.csv")
#     pass

# def data_preprocessing():
#     data_preprocessing("/opt/airflow/dataset/analysis_train.csv","/opt/airflow/dataset/test.csv")
#     pass

# config.py



with DAG('twitter_sentiment_analysis',
         start_date=datetime(2024,4,24),
         schedule_interval='@daily',
         catchup=False) as dag:

#dag = DAG('twitter_sentiment_analysis',start_date=datetime(2024,4,17), schedule_interval='@daily',catchup=False)


    check_files = PythonOperator(
        task_id='check_files',
        python_callable= files_exists,
    )

    data_eda = PythonOperator(
        task_id='data_eda',
        python_callable=data_analysis,
        op_args=[TRAIN_FILE, TEST_FILE]
    )

    data_prep = PythonOperator(
        task_id='data_prep',
        python_callable=data_preprocessing,
        op_args=[ANALYSIS_TRAIN_FILE, TEST_FILE]
    )

    data_train = PythonOperator(
        task_id='data_train',
        python_callable=data_training,
        op_args=[TRAIN_FILE, TEST_FILE]
    )

    check_files >> data_eda >> data_prep >> data_train

