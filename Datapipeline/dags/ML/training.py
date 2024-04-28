import os
import pandas as pd
import random
import spacy
from spacy.training import Example
import logging
from tqdm import tqdm

# Set up logging
log_directory = "/opt/airflow/logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

logging.basicConfig(filename=os.path.join(log_directory, 'data_training.log'),
                    level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def save_model(output_dir, nlp, new_model_name):
    output_dir ='/opt/airflow/dataset/working'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nlp.meta["name"] = new_model_name
    nlp.to_disk(output_dir)
    logging.info("Model saved to " + output_dir)

def load_model(model_path):
    logging.info("Loading model from " + model_path)
    return spacy.load(model_path)

def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        if [start, end, ent.label_] not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]:ent_array[0][1]] if ent_array else text
    return selected_text

def train_ner(train_data, output_dir, n_iter=20, model=None):
    if model is not None:
        nlp = spacy.load(output_dir)
        logging.info("Loaded model " + model)
    else:
        nlp = spacy.blank("en")
        logging.info("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner', last=True)
    else:
        ner = nlp.get_pipe('ner')

    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = spacy.util.minibatch(train_data, size=spacy.util.compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                examples = []
                for text, annotations in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annotations))
                nlp.update(examples, drop=0.5, losses=losses)
            logging.info("Iteration {} losses: {}".format(itn, losses))
    save_model(output_dir, nlp, 'st_ner')

def get_model_out_path(sentiment):
    '''
    Returns Model output path
    '''
    model_out_path = None
    if sentiment == 'positive':
        model_out_path = '/opt/airflow/dataset/model_pos'
    elif sentiment == 'negative':
        model_out_path = '/opt/airflow/dataset/model_neg'
    return model_out_path

def get_training_data(df_train,sentiment):
    '''
    Returns Trainong data in the format needed to train spacy NER
    '''
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))
    return train_data

# Data loading and processing
def data_training(train_file, test_file):
    logging.info("Starting data training process.")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    df_train['Num_words_text'] = df_train['text'].apply(lambda x: len(str(x).split()))
    logging.info("Number Of words in main Text in train set calculated.")
    df_train = df_train[df_train['Num_words_text'] >= 3]

    sentiment = 'positive'
    train_data = get_training_data(df_train, sentiment)
    model_path = get_model_out_path(sentiment)
    train_ner(train_data, model_path, n_iter=3, model=None)
    logging.info("Model trained for positive tweets.")

    sentiment = 'negative'
    train_data = get_training_data(df_train, sentiment)
    model_path = get_model_out_path(sentiment)
    train_ner(train_data, model_path, n_iter=3, model=None)
    logging.info("Model trained for negative tweets.")

    # Load models
    model_pos = load_model('/opt/airflow/dataset/model_pos')
    model_neg = load_model('/opt/airflow/dataset/model_neg')

    # Predicting
    selected_texts = []
    for index, row in df_test.iterrows():
        text = row.text
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        else:
            selected_texts.append(predict_entities(text, model_neg))
    df_test['selected_text'] = selected_texts

    # Example saving output to CSV, modify as necessary
    df_test.to_csv("/opt/airflow/dataset/predicted_data.csv", index=False)
    logging.info("Predicted data saved to CSV.")


# if __name__ == "__main__":
#     train_file = "train.csv"
#     test_file = "test.csv"
#     data_training(train_file, test_file)
