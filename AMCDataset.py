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

    def __init__(self, df:pd.DataFrame, **kwargs):
        '''
        :param df: the pandas data frame holding the data
        :param kwargs: parameters
        :return: nothing
        '''

        
        self.features_all = sorted(set(features))
        self.features_bin = bin_vector
        
        self.feature_info_num = []
        self.feature_info_cat = []
        self.feature_info_bin = []
        self.label_info    = label

        '''reduce locally to features we use'''
        df = pd_df[self.features_all]        
        
        num_labels  = []
        cat_labels  = []
        coded_cols  = []
        cat_distrib = []
        bin_vect    = []
        
        for col in df.columns:
            if col in bin_vector:
                self.feature_info_bin.append(col)
                '''convert/handle to binary vector, from index lixt'''
                for _, num_list in df[[col]].iterrows():
                    num_list_l = list(num_list)[0]                    
                    bin_vect.append( 
                        [1 if x in num_list_l else 0 for x in range(0,self.bin_vect_max)]
                    )
            elif is_numeric_dtype(df[col]):
                '''handle numerical data'''
                num_labels.append(col)
                #df[col] = df[col].astype('float64')
                df.loc[:, col] = df[col].astype('float64')
                self.feature_info_num.append(col)
            elif df[col].dtype.name == 'category':
                '''handle categroical data'''
                cat_labels.append(col)
                self.cat_sizes.append( len(df[col].dtype.categories) )
                coded_cols.append(df[col].cat.codes.values.tolist())
                try:
                    cat_distrib.append( 
                        self.__get_distribution(df[col].dtype.categories.values, df[col]) 
                    )
                except ValueError:
                    print('Error at cat col {}'.format(col))
                    raise
                    
                self.feature_info_cat.append(col)
            else:
                '''non float non categoric'''
                #print('skipping {}'.format(col))
        
        # DATA LISTS
        self.num_data  = [torch.tensor(l) for l in df[num_labels].values.tolist()]
        self.cat_data  = [torch.tensor(l) for l in zip(*coded_cols)]
        self.bin_vect  = [torch.tensor(l) for l in bin_vect]
        if 'cid' in label and 'cid' not in df.columns:
            '''cid is index'''
            sub_label = label.copy()
            sub_label.remove('cid')
            self.label     = [(a,b[0]) for a,b in zip(list(pd_df.index),pd_df[sub_label].values.tolist())]
        else:
            self.label     = pd_df[label].values.tolist()
        
        '''categorical distribustion and numerical max entries for normaliziation'''
        self.cat_distrib  = cat_distrib
        self.num_max   = self.num_data[0]
        for num_entry in self.num_data:
             self.num_max = torch.max(torch.cat((self.num_max, num_entry)).view(2,len(num_labels)), dim=0)[0]
       
        self.num_features = len(num_labels)
        self.cat_features = len(self.cat_sizes)
    
  
    
    def __len__(self):
        return len(self.label)
    
    
    def __repr__(self):
        repr_str  = 'features num: ' + ', '.join(self.feature_info_num) + '\n'
        repr_str += 'features cat: ' + ', '.join(self.feature_info_cat) + '\n'
        repr_str += 'features bin: ' + ', '.join(self.feature_info_bin) + '\n'
        repr_str += 'labels:   ' + ', '.join(self.label_info)
        return repr_str

    
    def __str__(self):
        return self.__repr__()
    
    
    def __getitem__(self, idx):
        x1, x2, y = self.num_data[idx], self.cat_data[idx], self.label[idx]
        bv = self.bin_vect[idx]
        
        if self.normalize:
            x1 = x1/self.num_max
        
        return (x1,x2,bv), y
    