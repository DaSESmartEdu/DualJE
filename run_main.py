import sys
import torch
import argparse
import random
from tqdm import trange
import time
import torch.nn as nn

from model.primal.transformer import transformer_Model
from model.dual.dual_joint_model_transformer import Title_Encoder, T2E


from JM.JM import JModel
from EM.EM_model import EModel

from metric import compute_avg_log_prob, calculate_reconstruction_probability, top_k_accuracy, result, mrr
from utils import *
import logging



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def dual(args, seed, primal_task_model, dual_task_model, title_model, vae_model,
         train_loader, valid_loader, title_vocab, primal_loss, device):
    
    setup_seed(seed)
    best_val_acc = float(0)
    early_stopping_patience = 3
    patience_counter = 0

    optimizer_primal = torch.optim.Adam(primal_task_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_dual = torch.optim.Adam(dual_task_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
 
    for epoch in trange(args.epochs, desc='Epoch'):
        epoch += 1
        primal_task_model.train()
        dual_task_model.train()

        train_dual_loss = 0.0
        train_primal_loss = 0.0
        train_acc_1s = []
        train_acc_2s = []
        train_acc_3s = []
        MRRs = []

        for idx, batch in enumerate(train_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            optimizer_primal.zero_grad()
            optimizer_dual.zero_grad()
            # dual train 

            # loss(T|API_E)
            p_titles_out = primal_task_model(user_APIsets_embs)
            batch_size, seq_len, num_titles = p_titles_out.size()
            p_titles_out = p_titles_out.reshape(-1, p_titles_out.shape[-1])
            user_title_ids = user_title_ids.reshape(-1)
            loss_1 = primal_loss(p_titles_out, user_title_ids)
            per_element_loss_1 = loss_1.reshape(batch_size, seq_len)
            masked_loss_1 = per_element_loss_1 * masks.float()
            loss_primal = masked_loss_1.sum() / masks.float().sum()
            #print('loss_primal:',loss_primal)
            
            #outputs = outputs.reshape(batch_size, seq_len, num_titles)
            user_title_ids = user_title_ids.reshape(batch_size, seq_len)
            # loss p(T)
            p_titles_out = p_titles_out.reshape(batch_size, seq_len, num_titles)
            title_loss = compute_avg_log_prob(p_titles_out, user_title_ids, title_model, device)
            #print('title_loss:', title_loss)
            title_loss = torch.neg(title_loss)
            #print('title_loss_n:',title_loss)


            # loss (API_E|T)
            APIs_embs_out = dual_task_model(user_title_ids, user_APIsetstypes)
            loss_dual = mse_loss(APIs_embs_out, user_APIsets_embs, masks)
            #loss_dual: tensor(10.5458, device='cuda:0', grad_fn=<MeanBackward0>)
            #print('loss_dual:', loss_dual)

            # loss p(API_E)
            recon_loss = calculate_reconstruction_probability(vae_model, APIs_embs_out, masks)
            #recon_loss: tensor(0.0005, device='cuda:0', grad_fn=<MeanBackward0>)
            #print('recon_loss:', recon_loss)
            
            # loss_duality
            loss_duality = (loss_primal + title_loss - loss_dual - recon_loss) ** 2
            #print('loss_duality:', loss_duality)
            



            loss_primal = loss_primal + loss_duality * args.alpha 
            loss_dual = loss_dual +  loss_duality * args.beta
            # print('reward:',reward)
            # break
            #primal_loss *= reward
            train_primal_loss += loss_primal * user_APIsets_embs.size(0)
            train_dual_loss += loss_dual * user_APIsets_embs.size(0)

            

            loss_primal.backward(retain_graph=True)
            grad_norm_primal = torch.nn.utils.clip_grad_norm_(primal_task_model.parameters(), args.clip)
            loss_dual.backward(retain_graph=False)
            grad_norm_primal = torch.nn.utils.clip_grad_norm_(dual_task_model.parameters(), args.clip)
            optimizer_primal.step()
            optimizer_dual.step()

            

            acc_1 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=1)
            train_acc_1s.append(acc_1.item())
            acc_2 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=2)
            train_acc_2s.append(acc_2.item())
            acc_3 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=3)
            train_acc_3s.append(acc_3.item())

            MRR = mrr(outputs, user_title_ids, masks)
            MRRs.append(MRR)

        train_primal_loss /= len(train_loader.dataset) 
        train_dual_loss /= len(train_loader.dataset)
        #print(len(train_acc_1s))
        top_1_acc = sum(train_acc_1s) / len(train_acc_1s)
        top_2_acc = sum(train_acc_2s) / len(train_acc_2s) 
        top_3_acc = sum(train_acc_3s) / len(train_acc_3s) 
        
        t_mrr = sum(MRRs) / len(MRRs)

        val_loss, precision, recall, F1, acc_1, acc_2, acc_3, val_mrr = evaluate(primal_task_model, valid_loader, primal_loss, device)
        
        logging.info(f'Epoch: {epoch},  Val Loss: {val_loss}, Val precision: {precision:.4f},  Val recall: {recall:.4f}, Val f1:{F1:.4f}, Val acc_1:{acc_1}, Val acc_2:{acc_2}, Val acc_3:{acc_3}, Train_MRR:{t_mrr}, Val_MRR:{val_mrr}')
        
        # 保存最佳模型
        if acc_1 > best_val_acc:
            best_val_acc = acc_1
            torch.save(primal_task_model.state_dict(), args.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

# 在测试集上评估
def evaluate(model, data_loader, primal_loss, device):
    model.eval()
    
    val_loss = 0.0  
    acc_1s = []
    acc_2s = []
    acc_3s = []
    
    MRRs = []

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            user_APIsets_embs = batch['user_APIsets_embs'].to(device)
            user_APIsetstypes = batch['user_APIsetstypes'].to(device)
            user_title_ids = batch['user_title_ids'].to(device)
            masks = batch['masks'].to(device)
            
            p_titles_out = model(user_APIsets_embs)
            batch_size, seq_len, num_titles = p_titles_out.size()
            p_titles_out = p_titles_out.reshape(-1, p_titles_out.shape[-1])  # [batch_size*seq_len, num_titles]
            user_title_ids = user_title_ids.reshape(-1) 
            
            loss = primal_loss(p_titles_out, user_title_ids)
            per_element_loss = loss.reshape(batch_size, seq_len)
            masked_loss = per_element_loss * masks.float()#.unsqueeze(-1)
            # masked_losses = per_element_losses * masks.float()#.unsqueeze(-1)

            # loss = masked_losses.sum() / masks.float().sum()
            loss = masked_loss.sum() / masks.float().sum()
            val_loss += loss.item() * user_APIsets_embs.size(0)
            
            p_titles_out = p_titles_out.reshape(batch_size, seq_len, num_titles)
            user_title_ids = user_title_ids.reshape(batch_size, seq_len)
                        
            acc_1 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=1)
            acc_1s.append(acc_1.item())
            acc_2 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=2)
            acc_2s.append(acc_2.item())
            acc_3 = top_k_accuracy(p_titles_out, user_title_ids, masks, k=3)
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


            


def mse_loss(predictions, APIs_embs, masks):
        # print('predictions:',predictions, type(predictions))
        # print('APIs_embs',APIs_embs, type(APIs_embs))
        # print(masks,type(masks))
        error = predictions - APIs_embs
        masked_error = error * masks.unsqueeze(-1)
        mse = torch.mean(torch.square(masked_error))
        return mse





def main(args):
    
    print("start loading data")
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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load primal task model
    primal_task_model = transformer_Model(args.APIsets_emsize, args.dropout, title_vocab)
    # primal_task_model.to(device)
    
    state_dict = torch.load(args.primal_model)
    primal_task_model.load_state_dict(state_dict)
    #print(primal_task_model)

    #load title model
    with open(args.JM, 'rb') as f:
        JM = torch.load(f)
    JM = JM.to(device)
    #print(JM)

    # load dual task model

    # dual task Base LSTM
    # dual_task_model = T2API_E(title_vocab, 6, args.emb_size, args.hidden_size, args.n_layers, args.dropout,
    #                 args.APIsets_emsize, args.bidirectional).to(device)
    # dual task Base transformer
    # dual_task_model = T2API_Ef(title_vocab, 6, args.hidden_size, args.dropout,
    #                 args.APIsets_emsize).to(device)
    
    # dual task joint and dual model
    title_Encoder = Title_Encoder(title_vocab, args.emb_size).to(device)
    dual_task_model = T2E(title_Encoder, args.emb_size,args.hidden_size, args.APIsets_emsize, args.dropout, 6).to(device)
    state_dict = torch.load(args.dual_model)
    dual_task_model.load_state_dict(state_dict)
    # print(dual_task_model)

    #load API model
    EM = EModel(args.APIsets_emsize, args.hidden_size, args.latent_size, args.n_layers).to(device)
    state_dict = torch.load(args.EM)
    EM.load_state_dict(state_dict)
    #print(vae_model)

    primal_loss = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    dual(args, args.seed, primal_task_model, dual_task_model, JM, EM,
         train_loader, val_loader, title_vocab, primal_loss,device)
    acc_report = evaluate(primal_task_model, test_loader, primal_loss, device)
    print(acc_report)  
        

    







if __name__ == '__main__':

    args = parse_args()
    #print(args)

    #load logging file
    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    
    # load model 
    # train_model = {}
    # score_model = {}

    # print(args)

    # dual(args)

    main(args)