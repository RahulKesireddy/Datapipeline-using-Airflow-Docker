#from airflow import Dataset
import re
import string
import numpy as np 
import random
import pandas as pd 
#import matplotlib.pyplot as plt
#import seaborn as sns
from collections import Counter
#from nltk.corpus import stopwords
from ML.dataunderstanding import data_understanding

train_file=r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv"
test_file=r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv"


def data_cleaning(train_file,test_file):
    
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    def clean_text(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    
    train['text'] = train['text'].apply(lambda x:clean_text(x))
    train['selected_text'] = train['selected_text'].apply(lambda x:clean_text(x))

    train.head()

    train['temp_list'] = train['selected_text'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in train['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']

    '''def remove_stopword(x):
        return [y for y in x if y not in stopwords.words('english')]
    train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))'''

    train.to_csv(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv", index=False)

data_cleaning(train_file,test_file)