''' Dataset Class '''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.sampler import BatchSampler
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import os, math


class AMCDataset(Dataset):
    ''' 
    AMCDataset holds all data for torch computation
    '''

    def __init__(self, df:pd.DataFrame, field_data:str = 'data', field_label:str = 'label'):
        self.max_seq_len  = max(x.shape[0] for x in df.data)
        #self.padding_list = [self.max_seq_len - x.shape[0] for x in df[field_data]]

        self.data    = [nn.ConstantPad1d((0, self.max_seq_len - d.shape[0]), 0)(d.transpose(1,0)).transpose(1,0) for d in df.data]

        #self.data    = [d for d in df[field_data]]
        self.labels  = [x for x in df[field_label]]
        self.indices = [idx for idx in df.index]
    
    
    def __len__(self):
        return len(self.labels)
    
    
    def __repr__(self):
        repr_str  = f"{len(self.labels)} items"
        return repr_str

    
    def __str__(self):
        return self.__repr__()
    
    
    def __getitem__(self, idx):
        data, idx, label = self.data[idx], self.indices[idx], self.labels[idx]
        #print('todo PADDING data')
        
        return data, (label, idx)
    