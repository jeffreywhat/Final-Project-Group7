import joblib
import pandas as pd

import re
import spacy
from nltk import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
def preprocess_data(X):
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
        if type(x) is str:
            for key in contractions:
                value = contractions[key]
                x = x.replace(key, value)
            return x
        else:
            return x

    # Function to lemmatize and remove stop words
    def lemmatize_and_remove_stop_words(text):
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc if token.text.lower() not in STOP_WORDS])
        return lemmatized_text

    # Function to lemmatize words
    def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # Preprocessing steps
    X = X.str.lower()
    X = X.apply(lambda x: cont_to_exp(x))
    X = X.apply(lambda x: re.sub('[^A-Z a-z 0-9-]+', ' ', x))
    X = X.apply(lambda x: re.sub(r'\b\d+\b', '', x))
    X = X.apply(lambda x: re.sub(r'<.*?>', '', x))
    X = X.apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))
    X = X.apply(lambda x: " ".join(x.split()))
    X = X.apply(lambda x: " ".join([t for t in x.split() if t not in STOP_WORDS]))
    X = X.apply(lambda text: lemmatize_words(text))
    X= X.str.strip()

    return X
def metrics(prediction, actual):
    print('Confusion_matrix \n', confusion_matrix(actual, prediction))
    print('\nAccuracy:', accuracy_score(actual, prediction))
    print('\nclassification_report\n')
    print(classification_report(actual, prediction))

# Loading Dataset
df_train = pd.read_csv('nlpproject/train.csv')
print(df_train.head(10))
columns_drop = ['id','keyword','location']

df_train = df_train.drop(columns=columns_drop)

print(df_train['target'].value_counts())
X_train, X_test,Y_train , Y_test = train_test_split(df_train['text'],df_train['target'], test_size=0.2)
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

print(X_train.head(10))



tf_vectorizer = TfidfVectorizer(max_features=10000,ngram_range=(1,2))

# convert into tfidf
xtrain_tf = tf_vectorizer.fit_transform(X_train)
xtest_tf = tf_vectorizer.transform(X_test)
vectorizer_filename = 'tfidf_vectorizer.joblib'
joblib.dump(tf_vectorizer, vectorizer_filename)

# Naive bayes model prediction
clb_tf = MultinomialNB().fit(xtrain_tf, Y_train)
predicted = clb_tf.predict(xtest_tf)
metrics(predicted, Y_test)
model_filename = 'multinomial_nb_model.joblib'
joblib.dump(clb_tf, model_filename)

print(f"Model saved to {model_filename}")