import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

# Set up logging
log_directory = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_analysis.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

train_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\train.csv"
test_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv"

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def data_analysis(train_file, test_file):
    logging.info("Starting data analysis process.")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    logging.info(f"Data loaded. Train shape: {train.shape}, Test shape: {test.shape}")

    train.info()
    train.dropna(inplace=True)
    test.info()

    # Plotting the sentiment count
    plt.figure(figsize=(12,6))
    sns.countplot(x='sentiment', data=train)
    plt.savefig(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\sentiment_count.png")
    plt.clf()
    logging.info("Sentiment count plot saved.")

    results_jaccard = []
    for ind, row in train.iterrows():
        sentence1 = row.text
        sentence2 = row.selected_text
        jaccard_score = jaccard(sentence1, sentence2)
        results_jaccard.append([sentence1, sentence2, jaccard_score])

    jaccard_df = pd.DataFrame(results_jaccard, columns=["text", "selected_text", "jaccard_score"])
    train = train.merge(jaccard_df, how='outer')
    logging.info("Jaccard scores calculated and merged.")

    # KDE plots
    plt.figure(figsize=(12,6))
    p1 = sns.kdeplot(train['Num_words_ST'], shade=True, color="r").set_title('Kernel Distribution of Number Of words')
    p2 = sns.kdeplot(train['Num_word_text'], shade=True, color="b")
    plt.savefig(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\word_distribution.png")
    plt.clf()
    logging.info("Word distribution plot saved.")

    plt.figure(figsize=(12,6))
    sns.kdeplot(train[train['sentiment'] == 'positive']['difference_in_words'], shade=True, color="b").set_title('Kernel Distribution of Difference in Number Of words')
    sns.kdeplot(train[train['sentiment'] == 'negative']['difference_in_words'], shade=True, color="r")
    plt.savefig(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\word_difference_distribution.png")
    plt.clf()
    logging.info("Word difference distribution plot saved.")

    plt.figure(figsize=(12,6))
    sns.kdeplot(train[train['sentiment'] == 'positive']['jaccard_score'], shade=True, color="b").set_title('KDE of Jaccard Scores across different Sentiments')
    sns.kdeplot(train[train['sentiment'] == 'negative']['jaccard_score'], shade=True, color="r")
    plt.legend(labels=['positive', 'negative'])
    plt.savefig(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\jaccard_distribution.png")
    plt.clf()
    logging.info("Jaccard score distribution plot saved.")

    k = train[train['Num_word_text'] <= 2]
    k.groupby('sentiment').mean()['jaccard_score']
    logging.info(f"Mean Jaccard scores for short texts computed.")

    train.to_csv(r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\understanding_train.csv", index=False)
    logging.info("Processed data saved as CSV.")

if 'AIRFLOW_HOME' not in os.environ:
    data_analysis(train_file, test_file)
    logging.info("Data analysis completed successfully.")
