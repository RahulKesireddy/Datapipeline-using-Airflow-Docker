import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
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

    def remove_stopword(x):
        return [y for y in x if y not in stopwords.words('english')]
    train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))

    top = Counter([item for sublist in train['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp = temp.iloc[1:,:]
    temp.columns = ['Common_words','count']

    train['temp_list1'] = train['text'].apply(lambda x:str(x).split()) #List of words in every row for text
    train['temp_list1'] = train['temp_list1'].apply(lambda x:remove_stopword(x)) #Removing Stopwords

    top = Counter([item for sublist in train['temp_list1'] for item in sublist])
    temp = pd.DataFrame(top.most_common(25))
    temp = temp.iloc[1:,:]
    temp.columns = ['Common_words','count']
    logging.info("Top common words computed.")

    Positive_sent = train[train['sentiment']=='positive']
    Negative_sent = train[train['sentiment']=='negative']
    Neutral_sent = train[train['sentiment']=='neutral']

    #MosT common positive words
    top = Counter([item for sublist in Positive_sent['temp_list'] for item in sublist])
    temp_positive = pd.DataFrame(top.most_common(20))
    temp_positive.columns = ['Common_words','count']
    logging.info("Most common positive words")

    #MosT common negative words
    top = Counter([item for sublist in Negative_sent['temp_list'] for item in sublist])
    temp_negative = pd.DataFrame(top.most_common(20))
    temp_negative = temp_negative.iloc[1:,:]
    temp_negative.columns = ['Common_words','count']
    logging.info("Most common negative words")

    #MosT common Neutral words
    top = Counter([item for sublist in Neutral_sent['temp_list'] for item in sublist])
    temp_neutral = pd.DataFrame(top.most_common(20))
    temp_neutral = temp_neutral.loc[1:,:]
    temp_neutral.columns = ['Common_words','count']
    logging.info("Most common neutral words")

    raw_text = [word for word_list in train['temp_list1'] for word in word_list]

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
    
    Unique_Positive= words_unique('positive', 20, raw_text)
    print("The top 20 unique words in Positive Tweets are:")
    logging.info("Top Positive Tweets")

    Unique_Negative= words_unique('negative', 10, raw_text)
    print("The top 10 unique words in Negative Tweets are:")
    logging.info("Top Negative Tweets")

    Unique_Neutral= words_unique('neutral', 10, raw_text)
    print("The top 10 unique words in Neutral Tweets are:")
    logging.info("Top Neutral Tweets")

    train.to_csv(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv", index=False)
    logging.info("Cleaned data saved to CSV.")

if 'AIRFLOW_HOME' not in os.environ:
    data_cleaning(train_file, test_file)
    logging.info("Data cleaning script completed without errors.")
