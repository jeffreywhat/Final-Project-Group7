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

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['combined_text'], df['target'], test_size=0.2, random_state=999
)


train_texts = train_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True)

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, train_labels)
val_dataset = TweetDataset(val_encodings, val_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

model_path = "distilbert_model.pth"

trainer.train()

torch.save(model.state_dict(), model_path)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.load_state_dict(torch.load(model_path))



