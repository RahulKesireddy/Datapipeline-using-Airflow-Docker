import os
import re
import string
import numpy as np
import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
from spacy.util import minibatch, compounding
from spacy.training import Example
from collections import Counter
import logging

log_directory = "/opt/airflow/logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def data_training(train_file,test_file):
    logging.info("Starting data training process.")

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set
    logging.info("Number Of words in main Text in train set")
    
    df_train = df_train[df_train['Num_words_text']>=3]

    def save_model(output_dir, nlp, new_model_name):    
     output_dir = f'../working/{output_dir}'
     if output_dir is not None:        
         if not os.path.exists(output_dir):
             os.makedirs(output_dir)
         nlp.meta["name"] = new_model_name
         nlp.to_disk(output_dir)
         logging.info("saved model")

    def train(train_data, output_dir, n_iter=20, model=None):
    
     if model is not None:
         nlp = spacy.load(output_dir)  # load existing spaCy model
         logging.info("Loaded model")
     else:
         nlp = spacy.blank("en")  # create blank Language class
         logging.info("Created blank 'en' model")
    
     # create the built-in pipeline components and add them to the pipeline
     # nlp.create_pipe works for built-ins that are registered with spaCy
     if "ner" not in nlp.pipe_names:
         ner = nlp.create_pipe("ner")
         nlp.add_pipe('ner', last=True)
     # otherwise, get it so we can add labels
     else:
         ner = nlp.get_pipe("ner")
    
     # add labels
     for _, annotations in train_data:
         for ent in annotations.get("entities"):
             ner.add_label(ent[2])

     # get names of other pipes to disable them during training
     other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
     with nlp.disable_pipes(*other_pipes):  # only train NER
         # sizes = compounding(1.0, 4.0, 1.001)
         # batch up the examples using spaCy's minibatch
         if model is None:
             nlp.begin_training()
         else:
             nlp.resume_training()


         for itn in tqdm(range(n_iter)):
             random.shuffle(train_data)
             batches = minibatch(train_data, size=compounding(4.0, 500.0, 1.001))    
             losses = {}
             for batch in batches:
                 #texts, annotations = zip(*batch)

                 examples = []
                 for text,annotations in batch:
                     entities = annotations.get('entities', [])
                     example = Example.from_dict(nlp.make_doc(text), {"entities": entities})
                     examples.append(example)

                 nlp.update(examples,  
                             drop=0.5,   
                             losses=losses, 
                             )
             logging.info("Losses")
     save_model(output_dir, nlp, 'st_ner')

    def get_model_out_path(sentiment):
     '''
     Returns Model output path
     '''
     model_out_path = None
     if sentiment == 'positive':
         model_out_path = 'models/model_pos'
     elif sentiment == 'negative':
         model_out_path = 'models/model_neg'
     return model_out_path
    
    def get_training_data(sentiment):
        
        #Returns Trainong data in the format needed to train spacy NER
        
        train_data = []
        for index, row in df_train.iterrows():
            if row.sentiment == sentiment:
                selected_text = row.selected_text
                text = row.text
                start = text.find(selected_text)
                end = start + len(selected_text)
                train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
        
        return train_data


    #TRAINING MODELS FOR POSITIVE AND NEGATIVE TWEETS
    sentiment = 'positive'

    train_data = get_training_data(sentiment)
    model_path = get_model_out_path(sentiment)
    # For DEmo Purposes I have taken 3 iterations you can train the model as you want
    train(train_data, model_path, n_iter=3, model=None)
    logging.info("models trained for positive tweets")

    sentiment = 'negative'

    train_data = get_training_data(sentiment)
    model_path = get_model_out_path(sentiment)

    train(train_data, model_path, n_iter=3, model=None)
    logging.info("models trained for negative tweets")

    #PREDICTING WITH TRAINED MODEL
    def predict_entities(text, model):
        doc = model(text)
        ent_array = []
        for ent in doc.ents:
            start = text.find(ent.text)
            end = start + len(ent.text)
            new_int = [start, end, ent.label_]
            if new_int not in ent_array:
                ent_array.append([start, end, ent.label_])
        selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
        return selected_text
    
    selected_texts = []
    MODELS_BASE_PATH = '../input/tse-spacy-model/models/'

    if MODELS_BASE_PATH is not None:
        print("Loading Models  from ", MODELS_BASE_PATH)
        model_pos = spacy.load(MODELS_BASE_PATH + 'model_pos')
        model_neg = spacy.load(MODELS_BASE_PATH + 'model_neg')
        
        for index, row in df_test.iterrows():
            text = row.text
            output_str = ""
            if row.sentiment == 'neutral' or len(text.split()) <= 2:
                selected_texts.append(text)
            elif row.sentiment == 'positive':
                selected_texts.append(predict_entities(text, model_pos))
            else:
                selected_texts.append(predict_entities(text, model_neg))
        
    df_test['selected_text'] = selected_texts

    df_train['selected_text'] = df_test['selected_text']
    df_train.to_csv("/opt/airflow/dataset/trained_data.csv", index=False)
    logging.info("trained data saved to csv")

