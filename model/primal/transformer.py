import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

class transformer_Model(nn.Module):
    def __init__(self, APIsets_emsize, dropout, title2idx):
        super(transformer_Model, self).__init__()
        self.APIsets_emsize = APIsets_emsize
        self.dropout = nn.Dropout(dropout)
        self.title_size = len(title2idx)
        #self.bidirectional = bidirectional

        #self.set_embedding = nn.Linear(self.APIs_emb_size, self.APIs_emb_size)
        self.set_embedding = nn.Sequential(
                                nn.Linear(self.APIsets_emsize, self.APIsets_emsize * 2),
                                nn.ReLU(),
                                nn.Linear(self.APIsets_emsize * 2, self.APIsets_emsize),
                                nn.ReLU()
                            )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(200, 2, dim_feedforward=200, dropout=dropout),
            2
        )
        self.hidden2tag = nn.Linear(200, self.title_size)

    def forward(self, APIs):
        # APIs: [batch_size, seq_len,  APIsets_emsize]
        #batch_size, seq_len, n_APIs = APIs.size()
        
        # 掩码处理
        seq_len = APIs.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len)) == 0
        mask = mask.to(APIs.device)
        #print(mask)
        APIs_embedded = self.set_embedding(APIs)
        #print("APIs_embedded:",APIs_embedded.size())
        APIs_embedded = APIs_embedded.permute(1, 0, 2)
        outputs = self.transformer(APIs_embedded, mask = mask)
        #print("outputs:",outputs.size())
        outputs = outputs.permute(1, 0, 2)
        outputs = self.hidden2tag(outputs)
        return outputs
