import os
import re
import string
import numpy as np
import pandas as pd
from collections import Counter
import logging

# Set up logging
log_directory = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_cleaning.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

train_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv"
test_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv"

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def data_cleaning(train_file, test_file):
    logging.info("Starting data cleaning process.")
    
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    logging.info("Original Train Shape: %s, Test Shape: %s", train.shape, test.shape)

    train['text'] = train['text'].apply(clean_text)
    train['selected_text'] = train['selected_text'].apply(clean_text)

    logging.info("Text cleaning completed.")

    # Compute top common words
    train['temp_list'] = train['selected_text'].apply(lambda x: str(x).split())
    top = Counter([item for sublist in train['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words', 'count']
    logging.info("Top common words computed.")

    train.to_csv(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv", index=False)
    logging.info("Cleaned data saved to CSV.")

if 'AIRFLOW_HOME' not in os.environ:
    data_cleaning(train_file, test_file)
    logging.info("Data cleaning script completed without errors.")
