import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def top_k_accuracy(logits, target, masks, k):
    batch_size, seq_len, n_titles = logits.size()
    top_k_logits = torch.topk(logits, k=k, dim=-1).indices
    target_expanded = target.unsqueeze(-1).expand(-1, -1, k)
    masks_expanded = masks.unsqueeze(-1).expand_as(target_expanded).float()

    correct_in_top_k = torch.sum((top_k_logits == target_expanded).float() * masks_expanded)
    mask_sum = torch.sum(masks).item()

    return correct_in_top_k / mask_sum

import torch

def mrr(logits, target, masks):
    """
    计算 Mean Reciprocal Rank (MRR)
    
    参数:
    logits (torch.Tensor): 模型输出的 logit 值, 形状为 (batch_size, seq_len, n_titles)
    target (torch.Tensor): 正确答案的索引, 形状为 (batch_size, seq_len)
    masks (torch.Tensor): 有效的位置掩码, 形状为 (batch_size, seq_len)
    
    返回:
    mrr (float): Mean Reciprocal Rank 值
    """
    batch_size, seq_len, n_titles = logits.size()
    
    # 对 logits 进行排序,得到每个样本每个位置的预测结果排名
    ranks = torch.argsort(torch.argsort(logits, dim=-1, descending=True), dim=-1) + 1
    
    # 找到每个样本每个位置正确答案的排名
    target_ranks = torch.gather(ranks, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    
    # 计算每个样本每个位置的 Reciprocal Rank,并求平均
    reciprocal_ranks = 1.0 / target_ranks
    mrr = torch.sum(reciprocal_ranks * masks) / torch.sum(masks)
    
    return mrr.item()
    
def compute_avg_log_prob(outputs, title_ids, title_model, device):
    """
    args:
        outputs: tensor, shape [batch_size, seq_length, n_titles]
        title_model: object, title model
        labels: tensor, shape [batch_size, seq_length]
        device: torch.device

    returns:
        log_probs: tensor, scalar
                   aberage log-probability of title_sequence
    """
    #print('outputs:',outputs.size())
    with torch.no_grad():
        title_model.eval()
        batch_size, seq_len, n_titles = outputs.size()
        presiction_title = torch.argmax(outputs, dim=-1)
        title_input_t =  presiction_title.transpose(0,1).contiguous()
        title_input_t = title_input_t.to(device)
        hidden = title_model.init_hidden(batch_size)
        output_1, hidden_1 = title_model(title_input_t, hidden)
        probabilities = F.softmax(output_1, dim=-1)
        output_probs = torch.log(probabilities)
        avg_log_prob = output_probs.mean()  # 计算平均对数概率
        #avg_probs = torch.exp(avg_probs)
    return avg_log_prob


def calculate_reconstruction_probability(model, prediction, mask):

    with torch.no_grad():
        model.eval()
        recon_batch, _, _ = model(prediction)  # 使用模型进行重构
        
    reconstruction_loss = torch.mean((recon_batch - prediction) ** 2, dim=(1, 2))

    reconstruction_prob = torch.exp(-reconstruction_loss)
        
    return reconstruction_prob.mean()


def result(model, data_loader, device):
    all_pred_titles = []
    all_true_titles = []

    with torch.no_grad():
        model.eval()
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)

            p_titles_out =  model(user_APIsets_embs)

            # 根据mask过滤预测结果和标签
            pred_titles = torch.argmax(p_titles_out, dim=-1)
            true_titles = user_title_ids

            # 根据mask过滤预测结果和标签
            masked_pred_titles = pred_titles[masks == 1]
            masked_true_titles = true_titles[masks == 1]

            # 将预测结果和真实标签添加到总列表中
            all_pred_titles.append(masked_pred_titles)
            all_true_titles.append(masked_true_titles)
        

    # 将列表转换为NumPy数组
    all_pred_titles = torch.cat(all_pred_titles).cpu().numpy()
    all_true_titles = torch.cat(all_true_titles).cpu().numpy()

    # 计算指标
    #accuracy = accuracy_score(all_true_titles, all_pred_titles)
    precision = precision_score(all_true_titles, all_pred_titles, average='weighted', zero_division=0)
    recall = recall_score(all_true_titles, all_pred_titles, average='weighted', zero_division=0)
    f1 = f1_score(all_true_titles, all_pred_titles, average='weighted', zero_division=0)
    return precision, recall, f1
