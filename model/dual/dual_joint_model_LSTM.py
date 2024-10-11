import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# title encoder
class Title_Encoder(nn.Module):
    
    def __init__(self, title_vocab, title_emb_size):
        super(Title_Encoder, self).__init__()

        self.title_vocab_size = len(title_vocab)
        self.title_emb_size = title_emb_size
        self.title_embedding = nn.Embedding(self.title_vocab_size, self.title_emb_size)

    def forward(self, title_ids):
        # title_ids: [batch_size, seq_len]
        #print('title_ids:',title_ids,title_ids.size())
        t_embeddings = self.title_embedding(title_ids)  # [batch_size, title_len, title_emb]
        #print('t_embeddings:',t_embeddings)
        return t_embeddings

# title 2 APIset_embedding
class T2E(nn.Module):

    def __init__(self, title_encoder, title_emb_size, ttype_len, hidden_size, n_layers, dropout, APIs_emb_size, bidirectional):
        
        super(T2E, self).__init__()
        self.title_encoder = title_encoder
        self.title_emb_size = title_emb_size
        self.ttype_len = ttype_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.APIs_emb_size = APIs_emb_size
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        

        self.ttype_embedding = nn.Sequential(nn.Embedding(self.ttype_len, self.title_emb_size),
                                         nn.Linear(self.title_emb_size, self.title_emb_size)
                                        )

        self.lstm = nn.LSTM(self.title_emb_size, self.hidden_size, num_layers=self.n_layers
                            , bidirectional=self.bidirectional, batch_first=True)

        self.hidden2APIs = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.hidden_size, self.APIs_emb_size)
                                        )
        #self.act = nn.Tanh()
    def forward(self, title_ids, APIs_type=None):
        # title_ids: [batch_size, seq_len]
        # APIs_type: [batch_size, seq_len]
        batch_size, seq_len = title_ids.size()
        title_embedded = self.title_encoder(title_ids)  #[batch_size, seq_len, title_emb_size]

        if APIs_type is not None:
            ttype_embedded = self.ttype_embedding(APIs_type)  #[batch_size, seq_len, title_emb_size]
            title_embedded = title_embedded + ttype_embedded

        lstm_out, _ = self.lstm(title_embedded) #[batch_size, seq_len, hidden_size * 2]
        lstm_out = self.dropout(lstm_out)

        out = self.hidden2APIs(lstm_out) #[batch_size, seq_len, APIs_emb_size]
        return out

# title 2 APIset_type
class Classifier(nn.Module):
    
    def __init__(self, title_encoder, title_emb_size, ttype_len, hidden_size, n_layers, dropout, APIs_emb_size, bidirectional):
        super(Classifier, self).__init__()
        
        self.title_encoder = title_encoder
        self.title_emb_size = title_emb_size
        self.ttype_len = ttype_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.APIs_emb_size = APIs_emb_size
        self.bidirectional = bidirectional

        self.translate_APIemb = nn.Linear(200, self.title_emb_size)

        self.lstm_cla = nn.LSTM(self.title_emb_size, self.hidden_size, num_layers=self.n_layers,
                           bidirectional=True, batch_first= True)
        
        self.linear_class = nn.Linear(self.hidden_size *2, self.hidden_size)
        
        self.logits_layer_cla = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size * 2),
                                         nn.Linear(self.hidden_size * 2, self.hidden_size),
                                         nn.Linear(self.hidden_size, self.ttype_len)
                                        )

    def attention(self, input_tensor, masks):
        # 计算注意力权重
        attn_weights = F.softmax(input_tensor, dim=-1)
        
        # 将掩码应用于注意力权重
        masked_attn_weights = attn_weights * masks.unsqueeze(-1)
        
        # 归一化注意力权重
        masked_attn_weights = masked_attn_weights / (torch.sum(masked_attn_weights, dim=-2, keepdim=True) + 1e-8)
        
        # 应用注意力权重到输入张量
        attended_tensor = input_tensor * masked_attn_weights
    
        return attended_tensor 

    def forward(self, title_ids, masks, APIsets_emb=None):
        #print('masks:',masks)
        batch_size, seq_len = title_ids.size()
        title_embedded = self.title_encoder(title_ids)
        
        if APIsets_emb is not None:
           #print(APIsets_emb.size())
           APIsets_emb = self.translate_APIemb(APIsets_emb)
        #print("APIsets_emb:",APIsets_emb.size())
           title_embedded = APIsets_emb + title_embedded
        
        lstm_out, _ = self.lstm_cla(title_embedded)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear_class(lstm_out)

        attention_feature = self.attention(lstm_out, masks)
        #print("attention_feature:",attention_feature,attention_feature.size())
        logits = self.logits_layer_cla(attention_feature) 
        
        return logits

class DualModel:
    def __init__(self, t2e, classifier, learning_rate, weight_decay):
        
        self.t2e = t2e
        self.classifier = classifier
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer1 = torch.optim.Adam(self.t2e.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        self.optimizer2 = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(list(self.t2e.parameters()) + list(self.classifier.parameters()),lr=learning_rate, weight_decay=self.weight_decay)
    def mse_loss(self, predictions, APIs_embs, masks):
        # print('predictions:',predictions, type(predictions))
        # print('APIs_embs',APIs_embs, type(APIs_embs))
        # print(masks,type(masks))
        error = predictions - APIs_embs
        masked_error = error * masks.unsqueeze(-1)
        mse = torch.mean(torch.square(masked_error))
        return mse
    
    def classifier_loss(self, outputs, labels, masks):
   
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        masks_flat = masks.view(-1)

        # 只考虑未被掩码的位置
        active_loss = masks_flat > 0
        active_logits = outputs_flat[active_loss]
        active_labels = labels_flat[active_loss]

        # 计算交叉熵损失
        loss = F.cross_entropy(active_logits, active_labels, reduction='sum')

         # 计算平均损失
        num_active_tokens = masks_flat.sum()
        average_loss = loss / num_active_tokens

        return average_loss
    

    
    def joint_train(self, title_ids, ttypes, APIsets_emb, masks=None):
        self.t2e.train()
        self.classifier.train()
        
        self.optimizer.zero_grad()

        emb_logits = self.t2e(title_ids, ttypes)
        t2e_loss = self.mse_loss(emb_logits, APIsets_emb, masks)

        class_logits = self.classifier(title_ids, masks, APIsets_emb)
        class_loss = self.classifier_loss(class_logits, ttypes, masks)

        loss = t2e_loss + class_loss

        loss.backward()
        
        self.optimizer.step()

        
        return emb_logits, loss

    def dual_train(self, title_ids, ttypes, APIsets_emb, masks=None):
        
        #print(self.t2e)
        #print(self.classifier)
        self.t2e.train()
        self.classifier.train()

        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        
        class_logits = self.classifier(title_ids, masks, APIsets_emb)
        #print('class_logits:', class_logits, class_logits.size())
        predictions = torch.argmax(class_logits, dim=-1)
        class_loss = self.classifier_loss(class_logits, ttypes, masks)
        #print('class_loss:',class_loss)
        class_loss.backward(retain_graph=True)
        self.optimizer2.step()
        
        emb_logits = self.t2e(title_ids, predictions)
        #print('emb_logits:', emb_logits)
        t2e_loss = self.mse_loss(emb_logits, APIsets_emb, masks)

        t2e_loss.backward(retain_graph=False)
        self.optimizer1.step()
        
        #print('emb_logits:',emb_logits, emb_logits.size())
        
        
        #re_t2e_
        #print('t2e_loss:',t2e_loss)
        
        
        loss = t2e_loss + class_loss

        
        return emb_logits, loss
    
    def validate(self, title_ids, ttypes, APIsets_emb, masks=None):

        self.t2e.eval()
        self.classifier.eval()
        with torch.no_grad():
            APIs_type_prediction_logits = self.classifier(title_ids, masks)
            APIs_type_prediction = torch.argmax(APIs_type_prediction_logits, dim=2)
            APIs_emb_prediction = self.t2e(title_ids, APIs_type_prediction)
            # 计算模型2的输出
            

            # 计算损失
            loss1 = self.mse_loss(APIs_emb_prediction, APIsets_emb, masks)
            loss2 = self.classifier_loss(APIs_type_prediction_logits, ttypes, masks)
            total_loss = loss1 + loss2


        # self.t2e.train()
        # self.classifier.train()
        return APIs_emb_prediction, total_loss

    # def test(self, title_ids, ttypes, masks=None):
    #     self.t2e.eval()
    #     #self.classifier.eval()
    #     with torch.no_grad():
    #         # 计算模型1的输出
    #         APIs_emb_prediction = self.t2e(title_ids, ttypes)

    #         # 计算模型2的输出
    #         #output2 = self.model2(input)


    #     return APIs_emb_prediction

    # def prediction_step(self, title_ids, ttypes=None):
    #     APIs_emb_prediction = self.t2e(title_ids, ttypes)
    #     return APIs_emb_prediction




