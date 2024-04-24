import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from tqdm import tqdm
from spacy.util import minibatch, compounding
from collections import Counter
import logging

log_directory = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

train_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\cleaned_train.csv"
test_file = r"C:\Users\rahul\OneDrive\Documents\Datapipeline\dataset\test.csv"

def training_data(train_file,test_file):
    logging.info("Starting data training process.")

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    train['Num_words_text'] = train['text'].apply(lambda x:len(str(x).split())) #Number Of words in main Text in train set
    logging.info("Number Of words in main Text in train set")
    
    train = train[train['Num_words_text']>=3]

    def save_model(output_dir, nlp, new_model_name):    
     output_dir = f'../working/{output_dir}'
     if output_dir is not None:        
         if not os.path.exists(output_dir):
             os.makedirs(output_dir)
         nlp.meta["name"] = new_model_name
         nlp.to_disk(output_dir)
         print("Saved model to", output_dir)

    def train(train_data, output_dir, n_iter=20, model=None):
    
     if model is not None:
         nlp = spacy.load(output_dir)  # load existing spaCy model
         print("Loaded model '%s'" % model)
     else:
         nlp = spacy.blank("en")  # create blank Language class
         print("Created blank 'en' model")
    
     # create the built-in pipeline components and add them to the pipeline
     # nlp.create_pipe works for built-ins that are registered with spaCy
     if "ner" not in nlp.pipe_names:
         ner = nlp.create_pipe("ner")
         nlp.add_pipe(ner, last=True)
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
                 texts, annotations = zip(*batch)
                 nlp.update(texts,  # batch of texts
                             annotations,  # batch of annotations
                             drop=0.5,   # dropout - make it harder to memorise data
                             losses=losses, 
                             )
             print("Losses", losses)
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
