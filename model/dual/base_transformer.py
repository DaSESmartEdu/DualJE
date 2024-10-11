import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Transformer

class T2API_Ef(nn.Module):
    def __init__(self, title_vocab, ttype_len, hidden_size, dropout, APIs_emb_size):
        
        super(T2API_Ef, self).__init__()
        self.title_vocab = title_vocab
        self.ttype_len = ttype_len
        #self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.APIs_emb_size = APIs_emb_size
        self.dropout = nn.Dropout(dropout)
        
        self.title_embedding = nn.Embedding(len(self.title_vocab), self.hidden_size * 2)
        #self.ttype_embedding = nn.Embedding(ttype_len, self.hidden_size)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_size * 2, 2, dim_feedforward=self.hidden_size, dropout=dropout),
            2
        )
        self.hidden2APIs = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.hidden_size, self.APIs_emb_size)
                                        )
        #self.act = nn.Tanh()

    def forward(self, title_ids, APIs_type=None):
        # title_ids: [batch_size, seq_len]
        batch_size, seq_len = title_ids.size()
        title_embedded = self.title_embedding(title_ids)  #[batch_size, seq_len, hidden_size * 2]
        #print('title_embedded:',title_embedded.size())


        title_embedded = title_embedded.permute(1, 0, 2)
        title_embedded = self.transformer(title_embedded)
        title_embedded = title_embedded.permute(1,0,2)
        #print('title_embedded:',title_embedded.size())


        title_embedded = self.dropout(title_embedded)
        out = self.hidden2APIs(title_embedded) #[batch_size, seq_len, APIs_emb_size]

        return out