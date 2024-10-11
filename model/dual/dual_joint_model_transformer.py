import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Transformer
# title encoder
class Title_Encoder(nn.Module):
    
    def __init__(self, title_vocab, title_emb_size):
        #t_hidden:200 
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
    
    def __init__(self, title_encoder, title_emb_size, hidden_size, APIsets_emsize, dropout, ttype_len=None):
        super(T2E, self).__init__()
        
        self.title_encoder = title_encoder
        self.APIsets_emsize = APIsets_emsize 
        self.ttype_len = ttype_len
        self.title_emb_size = title_emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.title_hidden = nn.Linear(self.title_emb_size, self.hidden_size *2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_size * 2, 2, dim_feedforward=self.hidden_size, dropout=self.dropout),
            2
        )
        self.dropout_layer_T2E = nn.Dropout(self.dropout)

        self.ttype_embedding = nn.Embedding(self.ttype_len, self.title_emb_size)
        self.ttype_fc = nn.Linear(self.title_emb_size, self.hidden_size * 2)

        
        self.logits_layer_T2E = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),
                                         nn.LeakyReLU(),
                                         nn.Linear(self.hidden_size, self.APIsets_emsize)
                                        )
       
        #self.act = nn.Tanh()
        
        
    def forward(self, title_ids, APIs_type=None):
        # title_ids: [batch_size, title_len]
        #print('title_ids:',title_ids.size())
        batch_size, seq_len = title_ids.size()

        title_embedded = self.title_encoder(title_ids)  #[batch_size, seq_len, title_emb_size]
        #print('t2e title_embedded:', title_embedded)
        title_embedded = self.title_hidden(title_embedded)
        #print('title_embedded:',title_embedded.size())
        if APIs_type is not None:
            ttype_embedded = self.ttype_embedding(APIs_type)  #[batch_size, seq_len, title_emb_size *2]
            ttype_embedded = self.ttype_fc(ttype_embedded)  #[batch_size, seq_len, hidden_size *2]
            #print('ttype:',ttype_embedded.size())
            title_embedded = title_embedded + ttype_embedded
        #mask
        # mask = torch.triu(torch.ones(seq_len, seq_len)) == 0
        # mask = mask.to(title_ids.device)

        title_embedded = title_embedded.permute(1, 0, 2)
        title_embedded = self.transformer(title_embedded)
        title_embedded = title_embedded.permute(1,0,2)


        outputs = self.dropout_layer_T2E(title_embedded) # [batch_size, title_len, hidden_size *2]
        #print('outputs_1:', outputs)
        outputs = self.logits_layer_T2E(outputs) # [batch_size, title_len, APIsets_emsize]
        # outputs = self.act(outputs)
        # outputs = outputs * 4
        return outputs

# title 2 APIset_type
class Classifier(nn.Module):
    
    def __init__(self, title_encoder, title_emb_size, hidden_size, ttype_len, dropout):
        super(Classifier, self).__init__()
        
        self.title_encoder = title_encoder
        self.title_emb_size = title_emb_size
        self.hidden_size = hidden_size
        self.ttype_len = ttype_len
        self.dropout = nn.Dropout(dropout)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_size * 2, 2, dim_feedforward=self.hidden_size, dropout=dropout),
            2
        )

        self.translate_titleemb = nn.Linear(self.title_emb_size, self.hidden_size * 2)
        self.translate_APIemb = nn.Linear(200, self.hidden_size * 2)
        
        
        self.linear_class = nn.Linear(self.hidden_size * 2, self.hidden_size)
        #self.classfier_attention = MaskedAttention(self.hidden_size)
        
        

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
    def forward(self, title_ids, APIsets_emb=None, masks=None):
        #print('title_ids:',title_ids)
        batch_size, seq_len = title_ids.size()
        title_embedded = self.title_encoder(title_ids)  #[batch_size, seq_len, title_emb_size]
        #print('cla title_embedded:',title_embedded)
        title_embedded = self.translate_titleemb(title_embedded) #[batch_size, seq_len, hidden_size]
        if APIsets_emb is not None:
           APIsets_emb = self.translate_APIemb(APIsets_emb)#[batch_size, seq_len, hidden_size]

           title_embedded = title_embedded + APIsets_emb
        
        #mask
        # mask = torch.triu(torch.ones(seq_len, seq_len)) == 0
        # mask = mask.to(title_ids.device)

        title_embedded = title_embedded.permute(1, 0, 2)
        title_embedded = self.transformer(title_embedded)
        title_embedded = title_embedded.permute(1,0,2)
        
        title_embedded = self.dropout(title_embedded)
        #print('title_embedded:',title_embedded.size())
        #print(masks.size())
        title_embedded = self.linear_class(title_embedded)
        attention_feature = self.attention(title_embedded, masks)
        #print('attention_feature:',attention_feature)
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
    
    def loss_fun(self, outputs, user_APIsets_embs, masks):
    #print(outputs.size())
        diff = outputs - user_APIsets_embs

        masked_diff = diff * masks.unsqueeze(-1)

        loss = torch.sum(masked_diff**2) / torch.sum(masks)

        return loss
    
    def classifier_loss(self, outputs, labels, masks):
        # 将模型输出和标签展平为二维张量
        # print('outputs:',outputs,outputs.size())
        # print('labels:',labels,labels.size())
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
        #self.t2e.load_state_dict(title_encoder.state_dict())
        emb_logits = self.t2e(title_ids, ttypes)
        t2e_loss = self.loss_fun(emb_logits, APIsets_emb, masks)
        # #t2e_loss.backward(retain_graph=True)
        # t2e_loss.backward()
        # self.optimizer1.step()

        class_logits = self.classifier(title_ids, APIsets_emb, masks)
        class_loss = self.classifier_loss(class_logits, ttypes, masks)
        #class_loss.backward(retain_graph=False)
        # class_loss.backward()
        # self.optimizer2.step()
        
        # print('class_loss',class_loss)
        # print('t2e_loss', t2e_loss)
        
        loss = class_loss + t2e_loss 
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

        class_logits = self.classifier(title_ids, APIsets_emb, masks)
        predictions = torch.argmax(class_logits, dim=2)
        class_loss = self.classifier_loss(class_logits, ttypes, masks)
        class_loss.backward(retain_graph=True)
        self.optimizer2.step()

        emb_logits = self.t2e(title_ids, predictions)
        t2e_loss = self.loss_fun(emb_logits, APIsets_emb, masks)
        t2e_loss.backward(retain_graph=False)
        self.optimizer1.step()

        loss =  class_loss +  t2e_loss
        return emb_logits, loss
    
    def validate(self, title_ids, ttypes, APIsets_emb, masks=None):
        # 将模型1和模型2的梯度置零
        self.t2e.eval()
        self.classifier.eval()
        with torch.no_grad():
           
            APIs_type_prediction_logits = self.classifier(title_ids, APIsets_emb, masks)
            APIs_type_prediction = torch.argmax(APIs_type_prediction_logits, dim=2)
            APIs_emb_prediction = self.t2e(title_ids, APIs_type_prediction)
            
            
            #dual model
            # APIs_emb_prediction = self.t2e(title_ids, ttypes)
            # APIs_type_prediction = self.classifier(title_ids, APIsets_emb)

            # 计算损失
            loss1 = self.mse_loss(APIs_emb_prediction, APIsets_emb, masks)
            loss2 = self.classifier_loss(APIs_type_prediction_logits, ttypes, masks)
            total_loss = loss1 + loss2


        # self.t2e.train()
        # self.classifier.train()
        #return APIs_emb_prediction 
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





