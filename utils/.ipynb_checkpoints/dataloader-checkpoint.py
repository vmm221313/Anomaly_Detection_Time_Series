import pandas as po

import torch
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, file_path, args):
        
        if type(file_path) == str: 
            self.df = po.read_csv(file_path)
            self.data = (self.df['W'] - self.df['W'].mean())/self.df['W'].std()
            
        elif type(file_path) == list:
            df = po.DataFrame()
            for file in file_path:
                df_t = po.read_csv(file)
                df = po.concat([df, df_t], axis = 0, ignore_index = True)

            self.df = df.reset_index(drop = True)
            self.data = (self.df['0'] - self.df['0'].mean())/self.df['0'].std()
            
        #self.dates = self.df['Date']
        self.data = torch.tensor(self.data.values, dtype = torch.float)
        
        self.idx = 0
        self.train_seq_len = args.train_seq_len
        self.val_seq_len = args.val_seq_len
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):
        if self.idx + self.train_seq_len + self.val_seq_len > len(self.data):
            self.idx = 0
            raise StopIteration
        
        self.idx += 1 
        return (self.data[self.idx - 1: self.idx - 1 + self.train_seq_len], self.data[self.idx - 1 + self.train_seq_len: self.idx - 1 + self.train_seq_len + self.val_seq_len])



