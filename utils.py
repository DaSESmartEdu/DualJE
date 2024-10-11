from torch.utils.data import Dataset,DataLoader
import torch
from tqdm import tqdm
import json
import codecs
from argument import parse_args
import os
import pickle
from collections import OrderedDict
import numpy as np
from itertools import chain
# def read_data(data_path):
#     """
#     Read data from the file and return a list of tuples.
#     Each tuple contains APIs used by the user and the user's job title.
#     (
#         [
#           [API_1,API_2,...],
#           [API_1,API_2,...],
#           ...
#         ],
#         [
#             title1,
#             title2,
#             ...
#         ]
#     )
#     """
#     all_user_data = []
#     with open(data_path, 'r', encoding='utf-8') as file:
#         for line in tqdm(file):
#             user_API, user_title = [], []
#             line_dict = json.loads(line)
#             career_path = line_dict['career_path']
            
#             for career in career_path:
#                 repo_name = list(career.keys())[0]
#                 apis = career[repo_name]['apis']
#                 title = career['career']['title']['role']
                
#                 user_API.append(apis)
#                 user_title.append(title)
                
#             assert len(user_API) == len(user_title)
#             all_user_data.append((user_API, user_title))
            
#     return all_user_data

def read_data(data_path):
    """ 
    Function: 
        Read data from the given datas file path.
    Args:
        datapath: file path of the datas
    Returns:
        APIsets: a list of user_APIsets, each APIset is a list of API
        APIstypes: corresponding APIset Type for each APIset
        titles: corresponding title for each APIset
    Example:
        APIsets:[['<START>'],['torch','torch.nn','torch.nn.functional',...],['pandas','numpy',...],...,['<END>']]
        APIstypes:['<START>',1,2,1,1,...,'<END>']
        titles:['<START>',title1,title2,...,'<END>']
    """
    #APIsets, APIstypes, titles = [], [], []
    all_user_data = []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f:
            # user_APIsets = [['<START>']]
            # user_APIsetstypes = [6]
            # user_titles = ['<START>']

            user_APIsets, user_APIsetstypes, user_titles = [], [], []
            
            json_line = json.loads(line)
            user_APIsets += json_line['apis']
            user_APIsetstypes += json_line['APIs_tlabel']
            user_titles += json_line['sc_titles']

            # user_APIsets += [['<END>']]
            # user_APIsetstypes += [7]
            # user_titles += ['<END>']

            assert len(user_APIsets) == len(user_APIsetstypes) == len(user_titles)
            all_user_data.append((user_APIsets, user_APIsetstypes, user_titles))
    return all_user_data

def read_titles(file_path):
    """
    Read job titles from the file and return a list of titles.
    """
    # with open(file_path, 'r', encoding='utf-8') as file:
    #     titles = [line.strip() for line in file.readlines()]
    # return titles
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['title2id']

def build_API2idx(all_corpus):
    """
    Build a dictionary mapping each API to a unique integer index.
    """
    API_vocabs = [
        API
        for corpus in all_corpus
        for user_API, _ in corpus
        for API in user_API
    ]
    API_vocabs = [API  for user_API in API_vocabs for API in user_API]
    API_vocabs = list(set(API_vocabs))
    API_vocab2idx = {API: idx + 1 for idx, API in enumerate(API_vocabs)}
    API_vocab2idx.update({"0": 0})
    return API_vocab2idx

def build_idx2API(API_vocab2idx):
    API_idx2vocab = {idx:vocab for vocab,idx in API_vocab2idx.items()}
    return API_idx2vocab

def build_title2idx(titles_list):
    """
    Build a dictionary mapping each job title to a unique integer index.
    """
    title2idx = {title: idx + 1 for idx, title in enumerate(titles_list)}
    title2idx.update({"igner": 0})
    return title2idx

def read_pickle(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


class LSTM_Dataset(Dataset):
    """
    Define a Dataset for the text data for LSTM model.
    """
    def __init__(self, data, title_vocab, seq_len, n_APIs, API_emb_dict, API_repeated):
        """
        init function
        """
        super(LSTM_Dataset, self).__init__()
        self.data = data
        #self.API_vocab2idx = API2idx
        #self.idx2API_vocab = idx2API
        self.title_vocab = title_vocab
        self.seq_len = seq_len
        self.n_APIs = n_APIs
        self.API_emb_dict = API_emb_dict
        self.API_repeated = API_repeated
    
    def map_API_to_emb(self, user_APIsets):
        """
        Map each APIset to its embedding vector.
        """
        result = []
        for APIset in user_APIsets:
            APIset_embedding = np.zeros(200, dtype=np.float32)
            for API in APIset:
                API_embedding = self.API_emb_dict[API]
                # 在对应的维度上进行累加
                APIset_embedding += API_embedding  
            result.append(APIset_embedding) 
        return result

    def __getitem__(self, item):
        """
        Prepare the tensors for a given item from the data.
        """
        user_APIsets, user_APIsetstypes, user_titles = self.data[item]

        new_user_APIsets = []

        # Remove the repeated APIs in the APIset according API_repeated
        for APIset in user_APIsets:
            if self.API_repeated:
               new_APIset = APIset
            else:
               new_APIset = list(OrderedDict.fromkeys(APIset))
            
            if len(new_APIset) < self.n_APIs:
                new_user_APIsets.append(new_APIset)
            else:
                new_user_APIsets.append(new_APIset[:self.n_APIs])

        # Mix sequence length
        if len(new_user_APIsets) < self.seq_len:
            API_pad_list = ['<PAD>'] 
            title_pad = ['<PAD>']
            APIsettype_pad = [0]
            new_user_APIsets = new_user_APIsets + [API_pad_list] * (self.seq_len - len(new_user_APIsets))
            user_titles = user_titles + title_pad * (self.seq_len - len(user_titles))
            user_APIsetstypes = user_APIsetstypes + APIsettype_pad * (self.seq_len - len(user_APIsetstypes))
        else:
            new_user_APIsets = new_user_APIsets[:self.seq_len]
            user_titles = user_titles[:self.seq_len]
            user_APIsetstypes = user_APIsetstypes[:self.seq_len]
        
        # Add <START> and <END> tokens
        # new_user_APIsets = [['<START>']] + new_user_APIsets + [['<END>']]
        # user_titles = ['<START>'] + user_titles + ['<END>']
        # user_APIsetstypes = [6] + user_APIsetstypes + [7] 

        #Ensure the same length
        assert len(new_user_APIsets) == len(user_titles) == len(user_APIsetstypes)

        #convert title to idx and construct mask tensor
        user_title_ids = [self.title_vocab.get(title.strip()) for title in user_titles]
        #user_title_ids = user_title_ids[1:]
        user_title_ids = user_title_ids
        masks = [bool(t) for t in user_title_ids]

        user_title_ids = np.array(user_title_ids)
        masks = np.array(masks)
    
        user_APIsets_embs = self.map_API_to_emb(new_user_APIsets)
        #user_APIsets_embs = np.array(user_APIsets_embs)[:-1]
        user_APIsets_embs = np.array(user_APIsets_embs)
        user_APIsetstypes = [int(t) for t in user_APIsetstypes]
        user_APIsetstypes = np.array(user_APIsetstypes)[:-1]
        # return {
        #     "user_APIsets_embs": torch.from_numpy(user_APIsets_embs),
        #     "user_title_ids": torch.tensor(user_title_ids, dtype=torch.long),
        #     "user_APIsetstypes": torch.tensor(user_APIsetstypes, dtype=torch.long),
        #     "masks": torch.tensor(masks, dtype=torch.long)
        # }
        return {
            "user_APIsets_embs": torch.from_numpy(user_APIsets_embs).type(torch.float32),
            "user_title_ids": torch.from_numpy(user_title_ids).type(torch.long),
            "user_APIsetstypes": torch.from_numpy(user_APIsetstypes).type(torch.long),
            "masks": torch.from_numpy(masks).type(torch.long)
        }


    def __len__(self):
        return len(self.data)



def build_loader(data, title_vocab, API_emb_dict, seq_len, n_APIs, API_repeated,
                batch_size, shuffle, num_workers):
    """
    Build a DataLoader for the LSTM model.
    """
    dataset = LSTM_Dataset(data, title_vocab, seq_len, n_APIs, API_emb_dict, API_repeated)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def main(args):
    """
    The main function to run the program.
    """
    # load title vocab
    title_path = './vocab/title_vocab.json'
    title_vocab = read_titles(title_path)
    #print(title_vocab)

    # load API emb dict
    API_emb_path = './vocab/API_emb_2_PSE.pickle'
    API_emb_dict = read_pickle(API_emb_path)

    train_data_path = os.path.join('./data', 'train.txt')
    train_data = read_data(train_data_path)
    #print(train_data[0])
    # print(len(train_data))
    val_data_path = os.path.join('./data', 'val.txt')
    train_data = read_data(val_data_path)

    test_data_path = os.path.join('./data', 'test.txt')
    test_data = read_data(test_data_path)

    # train_dataset = LSTM_Dataset(train_data, title_vocab, args.seq_len, args.n_APIs, API_emb_dict, args.API_repeated)
    # print(train_dataset[230])
    #print(API_emb_dict)
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

    a = []

    for batch in train_loader:
        user_APIsets_embs, user_title_ids, user_APIsetstypes, masks = batch["user_APIsets_embs"], batch["user_title_ids"], batch['user_APIsetstypes'], batch['masks']
        print(user_APIsets_embs.size())
        print(user_title_ids.size())
        print(user_APIsetstypes.size())
        print(masks.size())
        # print(masks)
        #a += user_APIsetstypes.tolist()
        break
    # flattened_list = list(chain.from_iterable(a))
    # print(set(flattened_list))

if __name__ == "__main__":
    args = parse_args()
    main(args)