import random
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn

from model.dual.dual_task.base_transformer import T2API_Ef
from model.dual.dual_task.base_lstm import T2API_E
import logging
import os
import json


from utils import *
from torch.nn.functional import cosine_similarity


def loss_fun(outputs, user_APIsets_embs, masks):
    #print(outputs.size())
    diff = outputs - user_APIsets_embs

    masked_diff = diff * masks.unsqueeze(-1)

    loss = torch.sum(masked_diff**2) / torch.sum(masks)

    return loss

def com_MAE(outputs, user_APIsets_embs, masks):
    
    diff = torch.abs(outputs - user_APIsets_embs)
    masked_diff = diff[masks] 
    mae = masked_diff.mean()

    return mae

def com_cos(outputs, user_APIsets_embs, masks):
    cos_sim = cosine_similarity(outputs, user_APIsets_embs, dim=-1)
    masked_cos_sim = cos_sim * masks
    
    # 计算非零元素的数量
    num_nonzero = torch.sum(masks)

    mean_cos_sim = torch.sum(masked_cos_sim) / num_nonzero

    return mean_cos_sim

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
#模型训练
def train_model(model, seed, epochs, train_loader, valid_loader, optimizer, device):
    
    setup_seed(seed)
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in trange(epochs, desc='Epoch'):
        train_loss = 0.0
        cos_sims = []
        MAEs = []
        num_train_samples = 0
        model.train()
        
        for idx, batch in enumerate(train_loader):
            #user_title_ids, masks = batch["user_APIsets_embs"], batch["user_title_ids"], batch['user_APIsetstypes'], batch['masks']
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)

            outputs = model(user_title_ids)
            #torch.clamp(prediction, -n, n)
            #print('outputs:',outputs.size())

            # batch_size, seq_len, num_titles = outputs.size()
            
            # outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            # user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]

            loss = loss_fun(outputs, user_APIsets_embs, masks)
            # per_element_losses = loss.reshape(batch_size, seq_len)
            # # print('per_element_losses:',per_element_losses.size())
            # # print('masks.float().unsqueeze(-1):',masks.float().unsqueeze(-1).size())
            # masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            # loss = masked_losses.sum() / masks.float().sum()

            train_loss += loss.item() * user_APIsets_embs.size(0)
         
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #last time
            # outputs = outputs.reshape(batch_size, seq_len, num_titles)
            # user_title_ids = user_title_ids.reshape(batch_size, seq_len)
            cos_simility = com_cos(outputs, user_APIsets_embs, masks)
            cos_sims.append(cos_simility)

            mae = com_MAE(outputs, user_APIsets_embs, masks)
            MAEs.append(mae)
        train_loss /= len(train_loader.dataset)  
        cossim = sum(cos_sims) / len(cos_sims) 
        mae = sum(MAEs) / len(MAEs)
        #Validation

        val_loss, val_cossim, val_mae = evaluate(model, valid_loader, device)
        
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Cos: {val_cossim}, Val MAE: {val_mae}')
        
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
def evaluate(model, data_loader, device):
    model.eval()
    
    val_loss = 0.0  
    val_cos_sims = []
    val_MAEs = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            
            outputs = model(user_title_ids)
            # batch_size, seq_len, num_titles = outputs.size()
            
            # outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            # user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]
            
            # loss = criterion(outputs, user_title_ids)
            # per_element_losses = loss.reshape(batch_size, seq_len)
            
            # masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            #loss = masked_losses.sum() / masks.float().sum()
            loss = loss_fun(outputs, user_APIsets_embs, masks)
            val_loss += loss.item() * user_APIsets_embs.size(0)

            #last time
            # outputs = outputs.reshape(batch_size, seq_len, num_titles)
            # user_title_ids = user_title_ids.reshape(batch_size, seq_len)
            cos_simility = com_cos(outputs, user_APIsets_embs, masks)            
            val_cos_sims.append(cos_simility)

            mae = com_MAE(outputs, user_APIsets_embs, masks)
            val_MAEs.append(mae)
            

    val_loss /= len(data_loader.dataset)
    val_cossim = sum(val_cos_sims) / len(val_cos_sims) 
    val_mae = sum(val_MAEs) / len(val_MAEs)      
    return  val_loss, val_cossim, val_mae
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
        shuffle = False,
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
        shuffle = False,
        num_workers = args.num_workers
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T2API_Ef(title_vocab, 6, args.hidden_size, args.dropout,
                    args.APIsets_emsize).to(device)
    # model = T2API_E(title_vocab, 6, args.emb_size, args.hidden_size, args.n_layers, args.dropout,
    #                  args.APIsets_emsize, args.bidirectional).to(device)
    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('start training')
    train_model(model, args.seed, args.epochs, train_loader, val_loader, optimizer, device)
    acc_report = evaluate(model, test_loader, device)
    print(acc_report)
if __name__ == "__main__":
    # 配置日志记录器
    args = parse_args()
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    #print(args.num_workers)
    main(args)