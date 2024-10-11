import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class T2API_E(nn.Module):
    def __init__(self, title_vocab, ttype_len, emb_size, hidden_size, n_layers, dropout, APIs_emb_size, bidirectional):
        
        super(T2API_E, self).__init__()
        self.title_vocab = title_vocab
        self.emb_size = emb_size
        self.ttype_len = ttype_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.APIs_emb_size = APIs_emb_size
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        
        self.title_embedding = nn.Embedding(len(self.title_vocab), self.emb_size)
        #self.ttype_embedding = nn.Embedding(ttype_len, self.emb_size)
        #self.ttype_fc = nn.Linear(self.emb_size, self.hidden_size * 2)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, bidirectional= self.bidirectional, batch_first= True)
        self.hidden2APIs = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.hidden_size, self.APIs_emb_size)
                                        )
        #self.act = nn.Tanh()
    def forward(self, title_ids):
        # title_ids: [batch_size, seq_len]
        batch_size, seq_len = title_ids.size()
        title_embedded = self.title_embedding(title_ids)  #[batch_size, seq_len, temb_size]
        lstm_out, _ = self.lstm(title_embedded) #[batch_size, seq_len, hidden_size]


        lstm_out = self.dropout(lstm_out)
        out = self.hidden2APIs(lstm_out) #[batch_size, seq_len, APIs_emb_size]
        return out