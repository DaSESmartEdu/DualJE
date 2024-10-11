import random
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn

from model.dual.dual_task.dual_joint_model_transformer import DualModel,Title_Encoder,T2E, Classifier
import logging
import os
import json


from utils import *
from torch.nn.functional import cosine_similarity
# def calculate_top_k_accuracy(logits, targets, k):
#     values, indices = torch.topk(logits, k=k, sorted=True)
#     y = torch.reshape(targets, [-1, 1])
#     print(indices)
#     print(y)
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


#模型训练
def train_model(model, epochs, train_loader, valid_loader, device):
    
    best_val_loss = float('inf')
    early_stopping_patience = 10
    patience_counter = 0

    for epoch in trange(epochs, desc='Epoch'):
        train_loss = 0.0
        cos_sims = []
        MAEs = []
        num_train_samples = 0
        #model.train()
        
        for idx, batch in enumerate(train_loader):
            #user_title_ids, masks = batch["user_APIsets_embs"], batch["user_title_ids"], batch['user_APIsetstypes'], batch['masks']
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)

            #predictions, loss = model.joint_train(user_title_ids, user_APIsetstypes, user_APIsets_embs, masks)
            predictions, loss = model.dual_train(user_title_ids, user_APIsetstypes, user_APIsets_embs, masks)
            # print('predictions:', predictions)
            # print('loss:', loss)
            train_loss += loss.item() * user_APIsets_embs.size(0)
            #break
            #predictions = model.prediction_step(user_title_ids, user_APIsetstypes)
            cos_simility = com_cos(predictions, user_APIsets_embs, masks)
            cos_sims.append(cos_simility)

            mae = com_MAE(predictions, user_APIsets_embs, masks)
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
            torch.save(model.t2e.state_dict(), args.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

# 在测试集上评估
def evaluate(model, data_loader, device):
    #model.eval()
    
    val_loss = 0.0  
    val_cos_sims = []
    val_MAEs = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            #user_APIsetstypes = None
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            
            predictions, loss = model.validate(user_title_ids, user_APIsetstypes, user_APIsets_embs, masks)
            val_loss += loss.item() * user_APIsets_embs.size(0)
            
            #predictions = model.prediction_step(user_title_ids, user_APIsetstypes)
            #last time
            # outputs = outputs.reshape(batch_size, seq_len, num_titles)
            # user_title_ids = user_title_ids.reshape(batch_size, seq_len)
            cos_simility = com_cos(predictions, user_APIsets_embs, masks)
            val_cos_sims.append(cos_simility)

            mae = com_MAE(predictions, user_APIsets_embs, masks)
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
    title_encoder = Title_Encoder(title_vocab, args.emb_size).to(device)
    # print(title_encoder)
    t2e = T2E(title_encoder, args.emb_size, args.hidden_size, args.APIsets_emsize, args.dropout, 6).to(device)
    # batch_size = 16
    # seq_len = 20

    # input_data = torch.randint(0, 10, (batch_size, seq_len)).to(device)  # 随机生成输入数据
    #output = t2e(input_data)  # 进行编码器的前向传播
    #print(output)

    classifier = Classifier(title_encoder, args.emb_size, args.hidden_size, 6, args.dropout).to(device)
    # input_data1 = torch.randint(0, 10, (batch_size, seq_len,200)).to(device)
    # output = classifier(input_data,input_data1)
    model = DualModel(t2e, classifier, args.lr, args.weight_decay)


    #print(model.t2e)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # #criterion = nn.CrossEntropyLoss(reduction='none')
    # #weights = [0.00001, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    # #inverse_weights = 1.0 / torch.tensor(weights)
    # #class_weights = torch.Tensor(weights).to(device)
    # #criterion = FocalLoss(alpha=2., gamma=2., class_weights=class_weights)
    # criterion = nn.MSELoss(reduction='none')
    # print('start training')


    train_model(model, args.epochs, train_loader, val_loader, device)
    acc_report = evaluate(model, test_loader, device)
    print(acc_report)
if __name__ == "__main__":
    # 配置日志记录器
    args = parse_args()
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    #print(args.num_workers)
    main(args)