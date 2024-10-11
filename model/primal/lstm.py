import torch
import torch.nn as nn
import torch.optim as optim

class LSTM_Model(nn.Module):
    def __init__(self, APIsets_emsize, hidden_size, n_layers, dropout, 
                 title2idx):
        super(LSTM_Model, self).__init__()
        self.APIsets_emsize = APIsets_emsize
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.title_size = len(title2idx)
        
        self.set_embedding = nn.Sequential(
                                nn.Linear(self.APIsets_emsize, self.APIsets_emsize *2 ),
                                nn.ReLU(),
                                nn.Linear(self.APIsets_emsize * 2, self.APIsets_emsize),
                                nn.ReLU()
                                )

        self.lstm = nn.LSTM(self.APIsets_emsize, self.hidden_size, num_layers=self.n_layers
                            , batch_first=True)
        self.hidden2title = nn.Linear(self.hidden_size, self.title_size)

    def forward(self, APIs):
        # APIs [batch_size, seq_len, APIsets_emsize]

        APIs_embedded = self.set_embedding(APIs)
        outputs, hidden = self.lstm(APIs_embedded)
        outputs = self.dropout(outputs)
        outputs = self.hidden2title(outputs)
        return outputs
