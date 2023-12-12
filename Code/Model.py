import torch
from torch import nn

embedding_dim =300
output_dim =1
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
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