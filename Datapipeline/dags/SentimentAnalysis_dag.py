from airflow import DAG,Dataset
from datetime import datetime
from airflow.operators.python import PythonOperator
import numpy as np
import pandas as pd
from collections import Counter
#from nltk.corpus import stopwords
from ML.dataunderstanding import data_analysis
from ML.datacleaning import data_cleaning


def load_data():
    
    train=Dataset(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\train.csv")
    test=Dataset(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass

def dataanalysis():
    data_analysis(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv",r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass

def datacleaning():
    data_cleaning(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv",r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass



with DAG('twitter_sentiment_analysis',
         start_date=datetime(2024,4,17),
         schedule_interval='@daily',
         catchup=False) as dag:

#dag = DAG('twitter_sentiment_analysis',start_date=datetime(2024,4,17), schedule_interval='@daily',catchup=False)


    get_data = PythonOperator(
        task_id='get_data',
        python_callable= load_data,
    )

    data_eda = PythonOperator(
        task_id='data_eda',
        python_callable=dataanalysis
    )

    data_prep = PythonOperator(
        task_id='data_prep',
        python_callable=datacleaning
    )

    get_data >> data_eda >> data_prep

