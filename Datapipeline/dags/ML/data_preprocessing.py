import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from collections import Counter
import logging

# Set up logging
log_directory = "/opt/airflow/logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_cleaning.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

#train_file = "/opt/airflow/dataset/understanding_train.csv"
#test_file = "/opt/airflow/dataset/test.csv"

try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def data_preprocessing(analysis_train_file, test_file):
    logging.info("Starting data cleaning process.")
    
    train = pd.read_csv(analysis_train_file)
    test = pd.read_csv(test_file)
    
    logging.info("Original Train Shape: %s, Test Shape: %s", train.shape, test.shape)

    train['text'] = train['text'].apply(clean_text)
    train['selected_text'] = train['selected_text'].apply(clean_text)

    logging.info("Text cleaning completed.")

    # COMMON WORDS IN SELECTED TEXT COLUMN
    train['temp_list'] = train['selected_text'].apply(lambda x: str(x).split())
    top = Counter([item for sublist in train['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']
    fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', 
             width=700, height=700,color='Common_words')
    fig.write_image("/opt/airflow/dataset/top_common_words_in_selected_text.png")
    logging.info("Common words in selected text plot saved.")

    
    def remove_stopword(x):
        return [y for y in x if y not in stopwords.words('english')]
    train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))

    #AFTER REMOVING STOP WORDS, THE MOST COMMON WORDS IN SELECTED TEXT COLUMN
    top = Counter([item for sublist in train['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp = temp.iloc[1:,:]
    temp.columns = ['Common_words','count']
    fig=px.treemap(temp, path=['Common_words'], values='count', title='Tree of Most Common Words')
    fig.write_image("/opt/airflow/dataset/top_common_words.png")
    logging.info("Most Common Words in Selected text plot saved")



    train['temp_list1'] = train['text'].apply(lambda x:str(x).split()) #List of words in every row for text
    train['temp_list1'] = train['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords

    # MOST COMMON WORDS IN TEXT COLUMN
    top = Counter([item for sublist in train['temp_list1'] for item in sublist])
    temp = pd.DataFrame(top.most_common(25))
    temp = temp.iloc[1:,:]
    temp.columns = ['Common_words','count']
    fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Text', orientation='h', 
             width=700, height=700,color='Common_words')
    fig.write_image("/opt/airflow/dataset/most_common_words_in_text.png")
    logging.info("Most Common Words in Text plot saved")

    #MOST COMMON WORDS SENTIMENT WISE
    Positive_sent = train[train['sentiment']=='positive']
    Negative_sent = train[train['sentiment']=='negative']
    Neutral_sent = train[train['sentiment']=='neutral']

    #MosT common positive words
    top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
    temp_positive = pd.DataFrame(top.most_common(20))
    temp_positive.columns = ['Common_words','count']
    fig = px.bar(temp_positive, x="count", y="Common_words", title='Most Commmon Positive Words', orientation='h', 
             width=700, height=700,color='Common_words')
    fig.write_image("/opt/airflow/dataset/most_common_positive_words.png")
    logging.info("Most common positive words plot saved")

    #MosT common negative words
    top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])
    temp_negative = pd.DataFrame(top.most_common(20))
    temp_negative = temp_negative.iloc[1:,:]
    temp_negative.columns = ['Common_words','count']
    fig = px.treemap(temp_negative, path=['Common_words'], values='count',title='Tree Of Most Common Negative Words')
    fig.write_image("/opt/airflow/dataset/most_common_negative_words.png")
    logging.info("Most common negative words plot saved")

    #MosT common Neutral words
    top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])
    temp_neutral = pd.DataFrame(top.most_common(20))
    temp_neutral = temp_neutral.loc[1:,:]
    temp_neutral.columns = ['Common_words','count']
    fig = px.bar(temp_neutral, x="count", y="Common_words", title='Most Commmon Neutral Words', orientation='h', 
             width=700, height=700,color='Common_words')
    fig.write_image("/opt/airflow/dataset/most_common_neutral_words.png")
    logging.info("Most common neutral words plot saved")

    raw_text = [word for word_list in train['temp_list1'] for word in word_list]

    #UNIQUE WORDS
    def words_unique(sentiment,numwords,raw_words):
  
     allother = []
     for item in train[train.sentiment != sentiment]['temp_list1']:
         for word in item:
             allother .append(word)
     allother  = list(set(allother ))

     specificnonly = [x for x in raw_text if x not in allother]
    
     mycounter = Counter()
    
     for item in train[train.sentiment == sentiment]['temp_list1']:
         for word in item:
             mycounter[word] += 1
     keep = list(specificnonly)
    
     for word in list(mycounter):
         if word not in keep:
             del mycounter[word]
    
     Unique_words = pd.DataFrame(mycounter.most_common(numwords), columns = ['words','count'])
    
     return Unique_words
    
    #UNIQUE WORDS IN POSITIVE TWEETS
    Unique_Positive= words_unique('positive', 20, raw_text)
    fig = px.treemap(Unique_Positive, path=['words'], values='count',title='Tree Of Unique Positive Words')
    fig.write_image("/opt/airflow/dataset/unique_words_in_positive_tweets.png")
    logging.info("Unique words in positive tweets plot saved")

    #UNIQUE WORDS IN NEGATIVE TWEETS
    Unique_Negative= words_unique('negative', 10, raw_text)
    fig = px.treemap(Unique_Negative, path=['words'], values='count',title='Tree Of Unique Negative Words')
    fig.write_image("/opt/airflow/dataset/unique_words_in_negative_tweets.png")
    logging.info("Unique words in negative tweets plot saved")

    #UNIQUE WORDS IN NEUTRAL TWEETS
    Unique_Neutral= words_unique('neutral', 10, raw_text)
    fig = px.treemap(Unique_Neutral, path=['words'], values='count',title='Tree Of Unique Neutral Words')
    fig.write_image("/opt/airflow/dataset/unique_words_in_neutral_tweets.png")
    logging.info("Unique words in neutral tweets plot saved")

    train.to_csv("/opt/airflow/dataset/cleaned_train.csv", index=False)
    logging.info("Cleaned data saved to CSV.")

# if 'AIRFLOW_HOME' not in os.environ:
#     data_preprocessing(train_file, test_file)
#     logging.info("Data cleaning script completed without errors.")
