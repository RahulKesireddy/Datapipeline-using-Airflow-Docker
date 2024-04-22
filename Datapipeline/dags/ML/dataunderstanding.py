#from airflow import Dataset
import re
import string
import numpy as np 
import random
import pandas as pd 
#import matplotlib.pyplot as plt
#import seaborn as sns
from collections import Counter

def data_understanding(train_file,test_file):
    train=pd.read_csv(train_file)
    test=pd.read_csv(test_file)

    print(train.shape)
    print(test.shape)

    train.info()
    
    train.dropna(inplace=True)

    test.info()
    
    train.head()

    train.describe()

    train.groupby('sentiment').count()['text'].reset_index().sort_values(by='text',ascending=False)
   
    """plt.figure(figsize=(12,6))
    sns.countplot(x='sentiment',data=train)"""

    def jaccard(str1, str2): 
        a = set(str1.lower().split()) 
        b = set(str2.lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    results_jaccard=[]

    for ind,row in train.iterrows():
        sentence1 = row.text
        sentence2 = row.selected_text

        jaccard_score = jaccard(sentence1,sentence2)
        results_jaccard.append([sentence1,sentence2,jaccard_score])

    jaccard = pd.DataFrame(results_jaccard,columns=["text","selected_text","jaccard_score"])
    train = train.merge(jaccard,how='outer')

    train['Num_words_ST'] = train['selected_text'].apply(lambda x:len(str(x).split())) #Number Of words in Selected Text
    train['Num_word_text'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main text
    train['difference_in_words'] = train['Num_word_text'] - train['Num_words_ST'] #Difference in Number of words text and Selected Text

    train.head()

    """plt.figure(figsize=(12,6))
    p1=sns.kdeplot(train['Num_words_ST'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')
    p1=sns.kdeplot(train['Num_word_text'], shade=True, color="b")"""

    """plt.figure(figsize=(12,6))
    p1=sns.kdeplot(train[train['sentiment']=='positive']['difference_in_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')
    p2=sns.kdeplot(train[train['sentiment']=='negative']['difference_in_words'], shade=True, color="r")

    plt.figure(figsize=(12,6))
    p1=sns.kdeplot(train[train['sentiment']=='positive']['jaccard_score'], shade=True, color="b").set_title('KDE of Jaccard Scores across different Sentiments')
    p2=sns.kdeplot(train[train['sentiment']=='negative']['jaccard_score'], shade=True, color="r")
    plt.legend(labels=['positive','negative'])"""

    k = train[train['Num_word_text']<=2]

    k.groupby('sentiment').mean()['jaccard_score']

    k[k['sentiment']=='positive']

    train.to_csv(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv", index=False)

train_file=r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\train.csv"
test_file=r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv"
data_understanding(train_file,test_file)