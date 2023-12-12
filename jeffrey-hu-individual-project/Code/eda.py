import nltk
from nltk import WordNetLemmatizer
import re
import pandas as pd
import spacy
from spacy.lang.en import STOP_WORDS
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

path = "faketweets/train.csv"
df = pd.read_csv(path)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accu = accuracy_score(labels, preds)
    return {
        'accuracy': accu,
        'f1': f1,
    }

def preprocess_data(x):
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how does",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        " u ": " you ",
        " ur ": " your ",
        " n ": " and "}

    def cont_to_exp(x):
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x


    def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    x = x.lower()
    x = cont_to_exp(x)
    x = re.sub('[^A-Z a-z 0-9-]+', ' ', x)
    x = re.sub(r'\b\d+\b', '', x)
    x = re.sub(r'<.*?>', '', x)
    x = re.sub(r'https?://\S+|www\.\S+', '', x)
    x = " ".join(x.split())
    x = " ".join([t for t in x.split() if t not in STOP_WORDS])
    x = lemmatize_words(x)
    x = x.strip()

    return x



df['combined_text'] = df['keyword'].fillna('') + ' ' + df['location'].fillna('') + ' ' + df['text'].apply(preprocess_data)

print(df['combined_text'].head())

print(df['target'])
counts = df['target'].value_counts()
print(counts)

from collections import Counter

words = [word for sentence in df['combined_text'] for word in sentence.split()]
word_counts = Counter(words)
most_common_words = word_counts.most_common(20)
print(most_common_words)

import matplotlib.pyplot as plt

# Flatten the list of words in the combined_text
words = [word for sentence in df['combined_text'] for word in sentence.split()]

# Count the words and get the 10 most common
word_counts = Counter(words)
most_common_words = word_counts.most_common(10)
most_common_words = word_counts.most_common(15)[5:]

# Unzip the words and their counts
words, counts = zip(*most_common_words)

plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in the Disaster Tweet Dataset')
plt.xticks(rotation=45)
plt.show()