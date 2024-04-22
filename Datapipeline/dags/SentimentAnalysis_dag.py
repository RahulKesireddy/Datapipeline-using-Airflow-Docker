from airflow import DAG,Dataset
from datetime import datetime
from airflow.operators.python import PythonOperator
import numpy as np
import pandas as pd
from collections import Counter
#from nltk.corpus import stopwords
from ML.dataunderstanding import data_understanding
from ML.datacleaning import data_cleaning


def load_data():
    
    train=Dataset(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\train.csv")
    test=Dataset(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass

def dataunderstanding():
    data_understanding(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv",r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass

def datacleaning():
    data_cleaning(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv",r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv")
    pass



with DAG('twitter_sentiment_analysis',start_date=datetime(2024,4,17),schedule_interval='@daily',catchup=False) as dag:

#dag = DAG('twitter_sentiment_analysis',start_date=datetime(2024,4,17), schedule_interval='@daily',catchup=False)


 load_data = PythonOperator(
    task_id='load_data',
    python_callable= load_data,
 )

 dataunderstanding= PythonOperator(
    task_id='dataunderstanding',
    python_callable=dataunderstanding
 )

 datacleaning= PythonOperator(
    task_id='datacleaning',
    python_callable=datacleaning
 )

load_data >> dataunderstanding >> datacleaning
