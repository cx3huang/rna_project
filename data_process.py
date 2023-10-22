import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class RNADataset(Dataset):
    def __init__(self, df, mode='train', **kwargs):
        self.bp_mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.Lmax = 206

        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        '''TODO: implement dataset splitting'''
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        b = (df_2A3['SN_filter'].values > 0.5) & (df_DMS['SN_filter'].values > 0.5)
        df_2A3 = df_2A3.loc[b].reset_index(drop=True)
        df_DMS = df_DMS.loc[b].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        
        '''TODO: check if this implementation correctly returns sequences'''
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        
        self.snr_2A3 = df_2A3['signal_to_noise'].values
        self.snr_DMS = df_DMS['signal_to_noise'].values

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        seq = np.array([self.seq_map[s] for s in seq])
        
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        
        reactivity = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        reactivity_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        snr = torch.FloatTensor([self.snr_2A3[idx],self.snr_DMS[idx]])
        
        return {'seq':torch.from_numpy(seq), \
                'reactivity': reactivity, \
                'reactivity_err': reactivity_err, \
                'snr':snr}