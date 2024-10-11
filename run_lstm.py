import random
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from model.primal.lstm import LSTM_Model
import logging
import os
import json

from metric import top_k_accuracy, mrr, result

from utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



#模型训练
def train_model(model, seed, epochs, train_loader, valid_loader, optimizer, criterion, device):
    
    setup_seed(seed)
    best_val_loss = float('inf')
    early_stopping_patience = 5
    patience_counter = 0

    for epoch in trange(epochs, desc='Epoch'):
        train_loss = 0.0
        train_acc_1s = []
        train_acc_2s = []
        train_acc_3s = []
        MRRs = []
        num_train_samples = 0
        model.train()
        
        for idx, batch in enumerate(train_loader):
            #user_title_ids, masks = batch["user_APIsets_embs"], batch["user_title_ids"], batch['user_APIsetstypes'], batch['masks']
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)

            outputs = model(user_APIsets_embs)
            
            #print('outputs:',outputs.size()) #[batch_size, seq_len, num_titles]

            batch_size, seq_len, num_titles = outputs.size()
            
            outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]

            loss = criterion(outputs, user_title_ids)
            per_element_losses = loss.reshape(batch_size, seq_len)
            # print('per_element_losses:',per_element_losses.size())
            # print('masks.float().unsqueeze(-1):',masks.float().unsqueeze(-1).size())
            masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            loss = masked_losses.sum() / masks.float().sum()

            train_loss += loss.item() * user_APIsets_embs.size(0)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = outputs.reshape(batch_size, seq_len, num_titles)
            user_title_ids = user_title_ids.reshape(batch_size, seq_len)
            #last_class, last_title = last_time(outputs, user_title_ids, masks)
       
            acc_1 = top_k_accuracy(outputs, user_title_ids, masks, 1)
            train_acc_1s.append(acc_1.item())
            acc_2 = top_k_accuracy(outputs, user_title_ids, masks, 2)
            train_acc_2s.append(acc_2.item())
            acc_3 = top_k_accuracy(outputs, user_title_ids, masks, 3)
            train_acc_3s.append(acc_3.item())
            
            MRR = mrr(outputs, user_title_ids, masks)
            MRRs.append(MRR)

        train_loss /= len(train_loader.dataset)  
        top_1_acc = sum(train_acc_1s) / len(train_acc_1s)
        top_2_acc = sum(train_acc_2s) / len(train_acc_2s) 
        top_3_acc = sum(train_acc_3s) / len(train_acc_3s) 
        
        t_mrr = sum(MRRs) / len(MRRs) 
        #Validation

        val_loss, precision, recall, F1, acc_1, acc_2, acc_3, val_mrr = evaluate(model, valid_loader, criterion, device)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val precision: {precision:.4f},  Val recall: {recall:.4f}, Val f1:{F1:.4f}, Val acc_1:{acc_1}, Val acc_2:{acc_2}, Val acc_3:{acc_3}, Train_MRR:{t_mrr}, Val_MRR:{val_mrr}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

# 在测试集上评估
def evaluate(model, data_loader,criterion, device):
    model.eval()
    
    val_loss = 0.0  
    acc_1s = []
    acc_2s = []
    acc_3s = []
    
    MRRs = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            #print("user_APIsets_embs:",user_APIsets_embs.size())
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            
            outputs = model(user_APIsets_embs)
            batch_size, seq_len, num_titles = outputs.size()
            
            outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]
            
            loss = criterion(outputs, user_title_ids)
            per_element_losses = loss.reshape(batch_size, seq_len)
            
            masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            loss = masked_losses.sum() / masks.float().sum()
            
            val_loss += loss.item() * user_APIsets_embs.size(0)

            outputs = outputs.reshape(batch_size, seq_len, num_titles)
            user_title_ids = user_title_ids.reshape(batch_size, seq_len)

            #last_class, last_title = last_time(outputs, user_title_ids, masks)

            acc_1 = top_k_accuracy(outputs, user_title_ids, masks, 1)
            acc_1s.append(acc_1.item())
            acc_2 = top_k_accuracy(outputs, user_title_ids, masks, 2)
            acc_2s.append(acc_2.item())
            acc_3 = top_k_accuracy(outputs, user_title_ids, masks, 3)
            acc_3s.append(acc_3.item())
            
            MRR = mrr(outputs, user_title_ids, masks)
            MRRs.append(MRR)

    val_loss /= len(data_loader.dataset)
    precision, recall, f1 = result(model, data_loader,device)
    top_1_acc = sum(acc_1s) / len(acc_1s)
    top_2_acc = sum(acc_2s) / len(acc_2s) 
    top_3_acc = sum(acc_3s) / len(acc_3s)
    val_mrr = sum(MRRs) / len(MRRs) 
    return  val_loss, precision, recall, f1, top_1_acc, top_2_acc, top_3_acc, val_mrr
def main(args):
    
    print("start loading realted data")
    #load title_vocab
    title_vocab = read_titles(args.title_dict)

    #load API Emb Dict
    API_emb_dict = read_pickle(args.API_emb_dict)

    #load data
    train_data_path = os.path.join(args.data, 'train.txt')
    train_data = read_data(train_data_path)

    val_data_path = os.path.join(args.data, 'val.txt')
    val_data = read_data(val_data_path)

    test_data_path = os.path.join(args.data, 'test.txt')
    test_data = read_data(test_data_path)

    train_loader = build_loader(
        data = train_data,
        title_vocab = title_vocab,
        API_emb_dict = API_emb_dict,
        seq_len = args.seq_len,
        n_APIs = args.n_APIs,
        API_repeated = args.API_repeated,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )

    val_loader = build_loader(
        data = val_data,
        title_vocab = title_vocab,
        API_emb_dict = API_emb_dict,
        seq_len = args.seq_len,
        n_APIs = args.n_APIs,
        API_repeated = args.API_repeated,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )

    test_loader = build_loader(
        data = test_data,
        title_vocab = title_vocab,
        API_emb_dict = API_emb_dict,
        seq_len = args.seq_len,
        n_APIs = args.n_APIs,
        API_repeated = args.API_repeated,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.num_workers
    )
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LSTM_Model(args.APIsets_emsize, args.hidden_size, args.n_layers, args.dropout,
                       title_vocab).to(device)
    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    print('start training')
    train_model(model, args.seed, args.epochs, train_loader, val_loader, optimizer, criterion, device)
    acc_report = evaluate(model, test_loader, criterion, device)
    print(acc_report)
if __name__ == "__main__":
    # 配置日志记录器
    args = parse_args()
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    #print(args.num_workers)
    main(args)