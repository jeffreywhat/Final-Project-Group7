import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from torch import nn
import joblib
import pickle
from torch.nn.functional import softmax
from PreProcess import preprocess_data, tokenize_text

device = "cpu"


with open('onehot_dicts.pkl', 'rb') as file:
    loaded_onehot_dict = pickle.load(file)
with open('embedding_matrix.pkl', 'rb') as file:
    print("going into")
    loaded_embedding_matrix = pickle.load(file)
print("exited")
output_dim = 1
hidden_dim = 256
embedding_dim = 300
no_layers = 1
vocab_size = len(loaded_onehot_dict) +1
embedding_matrix = loaded_embedding_matrix


class TwitterClassification(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_matrix):
        super(TwitterClassification, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=no_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.hidden_dim*2,128)
        self.relu = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128,output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.view(batch_size, -1, self.hidden_dim*2)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_fc1(out)
        out = self.fc2(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros((self.no_layers*2, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers*2, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        return hidden


bilstm_model = TwitterClassification(no_layers, vocab_size, hidden_dim, loaded_embedding_matrix)
bilstm_model = bilstm_model.to(device)


naivebayes_model = joblib.load('multinomial_nb_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
distilbert_model.load_state_dict(torch.load('distilbert_model.pth', map_location=torch.device('cpu')))
distilbert_model.eval()
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

checkpoint = torch.load('best_modellatests.pth', map_location=torch.device('cpu'))
bilstm_model.load_state_dict(checkpoint['model_state_dict'])
#h = bilstm_model.init_hidden(1)
bilstm_model.eval()


def classifyw_distilbert(text, model, tokenizer_func):
    processed_text = preprocess_data(text)
    inputs = tokenizer_func(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = softmax(logits, dim=1)
        prediction = probabilities.argmax().item()
    return prediction


def classifyw_naivebayes(text, model):
    processed_text = preprocess_data(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    return prediction


def classifyw_bilstm(text, model):
    preprocessed_text = preprocess_data(text)
    tokenized_text = tokenize_text(preprocessed_text)
    input_tensor = torch.tensor(tokenized_text).unsqueeze(0).to(device)
    h = model.init_hidden(1)
    with torch.no_grad():
        output, _ = model(input_tensor, h)
        threshold = 0.5
        prediction = (output > threshold).long().item()
    return prediction


# Streamlit UI
st.title("Disaster Tweet Classification App")

# Model selection
model_option = st.selectbox("Select the model for classification:", ("Naive Bayes", "DistilBERT", "BiLSTM"))

# Text input
user_input = st.text_area("Enter text here:")

dis_link = "https://www.noaa.gov/"
med_link = "https://www.webmd.com/"

if st.button("Classify"):
    if user_input:
        if model_option == "Naive Bayes":
            prediction = classifyw_naivebayes(user_input, naivebayes_model)
        elif model_option == "DistilBERT":
            prediction = classifyw_distilbert(user_input, distilbert_model, tokenizer)
        else:
            prediction = classifyw_bilstm(user_input, bilstm_model)
        st.write(f"Prediction: {prediction}")
        if prediction == 1:
            st.write(f"The model classifies this tweet or text as a disaster."
                     f"\nThe disaster helpline is 1-800-985-5990 in the United States of America."
                     f"\nIf you feel that your safety is in immediate danger, do not hesitate to call 911."
                     f"\nServices like NOAA will show weather-related emergencies, "
                     f"and WebMD can provide help for medical emergencies.")
            st.write(f"[Click here for a link to NOAA.gov, for the latest weather disaster update.]({dis_link})")
            st.write(f"[Click here for WebMD, a resource to help search for procedures during medical emergencies.]({med_link})")
        else:
            st.write(f"The model does not classify this tweet as a disaster."
                     f"\nIf you feel that there is an error with this classification, "
                     f"please contact jeffreyhu149@gmail.com")
    else:
        st.write("Please enter text or a tweet.")



