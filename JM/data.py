import os
import torch
import json

import sys
# class Dictionary(object):
#     def __init__(self):
#         self.word2idx = {} # word: index
#         self.idx2word = [] # position(index): word

#     def add_word(self, word):
#         if word not in self.word2idx:
#             self.idx2word.append(word)
#             self.word2idx[word] = len(self.idx2word) - 1
#         return self.word2idx[word]

#     def __len__(self):
#         return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        #self.dictionary = Dictionary()
        # three tensors of title index  a title dict
        self.title_path = os.path.join(path, 'title_vocab.json')
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'val.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        
        #self.title = self.title_dict(os.path.join(path, 'titles.json'))
    
    def title_dict(self, title_path):
        assert os.path.exists(title_path)

        with open(title_path, 'r') as f:
             data = json.load(f)
        return data['title2id']
        

    def tokenize(self, path):
        print(path)
        assert os.path.exists(path)
        # Add words to the dictionary
        all_title = []
        title_dict = self.title_dict(self.title_path)
        with open(path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                user_title = []
                line_dict = json.loads(line)
                #print(line_dict.keys())
                career_path = line_dict['career_path']

                for career in career_path:
                    title = career['career']['title']['role']
                    user_title.append(title.strip())
                #print(user_title)
                #sys.exit()
                # line to list of token + eos
                titles = user_title
                all_title.append(titles)
                tokens += len(titles)

        # Tokenize file content
       
        ids = torch.LongTensor(tokens)
        token = 0
        for titles in all_title:
            for title in titles:
                ids[token] = title_dict[title]
                token += 1

        return ids
    
    