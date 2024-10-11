import random
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from EM_model import EModel
import logging
import os
import json

import sys
sys.path.append('..')
from argument import parse_args
from utils import *

def loss_fun(outputs, title_ids, masks, criterion):
    outputs = outputs.view(-1, outputs.size(-1))      # [batch_size * seq_len, num_classes]
    title_ids = title_ids.view(-1)                       # [batch_size * seq_len]

    raw_loss = criterion(outputs, title_ids)            # [batch_size * seq_len]

    # 将masks转换为与raw_loss相同的形状然后应用于raw_loss
    masks = masks.view(-1).float()                 # [batch_size * seq_len]
    masked_loss = raw_loss * masks                 # [batch_size * seq_len]

    # 通过某种方式（例如，平均值，总和等）合并损失值来计算最终的损失
    batch_loss = torch.sum(masked_loss)
    mask_count = torch.sum(masks)

    if mask_count > 0:
        # 如果mask_count大于0,我们通过总损失除以mask_count来获取平均损失，这主要是为了考虑到我们可能没有使用所有的序列长度
        batch_loss /= mask_count

    return batch_loss

# def calculate_top_k_accuracy(logits, targets, k):
#     values, indices = torch.topk(logits, k=k, sorted=True)
#     y = torch.reshape(targets, [-1, 1])
#     correct = (y == indices) * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
#     top_k_accuracy = torch.mean(correct) * k  # 计算最后的准确率
#     return top_k_accuracy

# def last_time(outputs, titles, masks):
#     batch_size, seq_len, num_classes = outputs.size()

#     #去掉 <START> <END>
#     # outputs = outputs[:, 1:seq_len-1, :]
#     # titles = titles[:, 1:seq_len-1]
#     # masks = masks[:, 1:seq_len-1]

#     # 确定每条数据的最后一个有效时刻的位置
#     last_valid_index = torch.sum(masks, dim=-1) - 1
    
#     #last time
#     last_class = outputs[torch.arange(outputs.size(0)), last_valid_index]  #[batch_size]
#     last_title = titles[torch.arange(titles.size(0)), last_valid_index]
#     #print('last_title:',last_title)
#     return last_class, last_title

#模型训练
def train_model(model, epochs, train_loader, valid_loader, optimizer, criterion, device):
    
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in trange(epochs, desc='Epoch'):
        train_loss = 0.0
        num_train_samples = 0
        model.train()
        
        for idx, batch in enumerate(train_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)

            outputs , mu, logvar = model(user_APIsets_embs, masks)
            
            #print('outputs:',outputs.size())
            #torch.autograd.set_detect_anomaly(True)
            # batch_size, seq_len, num_titles = outputs.size()
            
            # outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            # user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]

            # loss = criterion(outputs, user_title_ids)
            # per_element_losses = loss.reshape(batch_size, seq_len)
            # # print('per_element_losses:',per_element_losses.size())
            # # print('masks.float().unsqueeze(-1):',masks.float().unsqueeze(-1).size())
            # masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)
            #计算重构损失
            # loss_1 = masked_losses.sum() / masks.float().sum()
            loss = model.loss_function(outputs, user_APIsets_embs, mu, logvar, masks)
            #print('loss_2:',loss_2.size())
            #print(loss_2)
            #loss_2 = -loss_2.to(loss_1.device)
            
            #lambda_t = -0.01
            #loss = loss_1 + lambda_t * loss_2
            #print(-0.2 * loss_2)
            #loss = loss_1.clone()
            #loss = loss_1 + lambda_t * loss_2
            #print('loss:',loss)
            train_loss += loss.item()
         
            optimizer.zero_grad()
            loss.backward()
            # 关闭异常检测
            #torch.autograd.set_detect_anomaly(False)
            optimizer.step()
                        
       
           

        train_loss /= len(train_loader.dataset)  
 
        #Validation

        val_loss = evaluate(model, valid_loader, criterion, device)
        
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
        
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
def evaluate(model, data_loader, criterion, device):
    model.eval()
    
    val_loss = 0.0  
    
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            outputs , mu, logvar = model(user_APIsets_embs)
            
            # batch_size, seq_len, num_titles = outputs.size()
            
            # outputs = outputs.reshape(-1, outputs.shape[-1])  # [batch_size*seq_len, num_titles]
            # user_title_ids = user_title_ids.reshape(-1)  # [batch_size*seq_len]
            
            # loss = criterion(outputs, user_title_ids)
            # per_element_losses = loss.reshape(batch_size, seq_len)
            
            # masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            # loss = masked_losses.sum() / masks.float().sum()

            # 计算重构损失和KL散度损失
            # recon_loss = criterion(outputs, data)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss = model.loss_function(outputs, user_APIsets_embs, mu, logvar, masks)

            val_loss += loss.item()
            #print('loss:',loss)
            #val_loss += loss.item() * user_APIsets_embs.size(0)

            #last time
            # outputs = outputs.reshape(batch_size, seq_len, num_titles)
            # user_title_ids = user_title_ids.reshape(batch_size, seq_len)       
        val_loss /=   len(data_loader.dataset)  
    return  val_loss
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
    # for batch in train_loader:
    #     APIs_ids, title_ids, masks = batch["APIs_ids"], batch["title_ids"], batch["masks"]
    #     print(APIs_ids.size(), title_ids.size(), masks.size())
    #     break

    #title model
    # with open(args.title_model, 'rb') as f:
    #     title_model = torch.load(f)
    # title_model = title_model.to(device)
    #title_model.eval()
    #print(title_model)
    model = EModel(args.APIsets_emsize, args.hidden_size, args.latent_size, args.n_layers).to(device)
    
    #print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    # weights = [0.00001, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    # #inverse_weights = 1.0 / torch.tensor(weights)
    # class_weights = torch.Tensor(weights).to(device)
    # criterion = FocalLoss(alpha=2., gamma=2., class_weights=class_weights)
    #criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    print('start training')
    train_model(model, args.epochs, train_loader, val_loader, optimizer, criterion, device)
    acc_report = evaluate(model, test_loader, criterion, device)
    print(acc_report)

if __name__ == "__main__":
    # 配置日志记录器
    args = parse_args()
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    #print(args.num_workers)
    main(args)