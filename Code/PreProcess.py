import joblib
from spacy.lang.en.stop_words import STOP_WORDS
from nltk import WordNetLemmatizer
import nltk
import spacy
import re
from Model import TwitterClassification
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
import torch

import numpy as np
import pickle
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
def preprocess_data(text):
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



    # Function to lemmatize words
    def lemmatize_words(text):
        lemmatizer = WordNetLemmatizer()
        return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    # Preprocessing steps
    text = text.lower()
    text = cont_to_exp(text)
    text = re.sub('[^A-Z a-z 0-9-]+', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = " ".join(text.split())
    text = " ".join([t for t in text.split() if t not in STOP_WORDS])
    text = lemmatize_words(text)
    text = text.strip()
    return text

def tokenize_text(text):

    final_list = []
    for word in text.split():
        if word in loaded_onehot_dict.keys():
            print(word)
            final_list.append(loaded_onehot_dict[word])

    return final_list


with open('onehot_dicts.pkl', 'rb') as file:
    loaded_onehot_dict = pickle.load(file)

    
# # loading tfidf vectorizer
# loaded_vectorizer = joblib.load("tfidf_vectorizer.joblib")
# # loading naive bayes
# loaded_model = joblib.load("multinomial_nb_model.joblib")
#
#
#
#
# # this are required when create model
# with open('embedding_matrix.pkl', 'rb') as file:
#     print("going into")
#     loaded_embedding_matrix = pickle.load(file)
# print("exited")
# hidden_dim = 256
# no_layers = 1
# vocab_size = len(loaded_onehot_dict) +1
# embedding_matrix = loaded_embedding_matrix
#
# # loading Bilstm
# checkpoint = torch.load('best_modellatests.pth')
# model = TwitterClassification(no_layers, vocab_size, hidden_dim, embedding_matrix)
# model = model.to(device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
#
# # streamlit example when user puts the tweet in
# user_input = "Just happened a terrible car crash"
# processed_input = preprocess_data(user_input)
#
# # for naive bayes
# text_transformed = loaded_vectorizer.transform([processed_input])
# prediction_NB = loaded_model.predict(text_transformed)[0]
# print(prediction_NB)
#
#
# # for Bi lstm output
# tokenized_input = tokenize_text(processed_input)
# input_tensor = torch.from_numpy(np.array([tokenized_input]))
# input_tensor = input_tensor.to(device)
# with torch.no_grad():
#     output, _ = model(input_tensor, model.init_hidden(1))
#     prediction = torch.round(output).item()
#
# # Print the prediction
# if prediction == 1:
#     print("The text is related to a disaster.")
# else:
#     print("The text is not related to a disaster.")
