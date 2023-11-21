import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, KFold

class RNADataset(Dataset):
    def __init__(self, df, mode='train', fold=0, n_folds=4, **kwargs):
        self.bp_mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.Lmax = 206

        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        split = list(KFold(n_splits=n_folds, shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)

        # b = (df_2A3['SN_filter'].values > 0.5) & (df_DMS['SN_filter'].values > 0.5)
        # df_2A3 = df_2A3.loc[b].reset_index(drop=True)
        # df_DMS = df_DMS.loc[b].reset_index(drop=True)

        self.seq = df_2A3['sequence'].values
        

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        
        # self.snr_2A3 = df_2A3['signal_to_noise'].values
        # self.snr_DMS = df_DMS['signal_to_noise'].values


    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        seq = np.array([self.bp_mapping[s] for s in seq])

        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        
        seq = np.pad(seq,(0,self.Lmax-len(seq)))

        s2A3 = self.react_2A3[idx]
        s2A3[s2A3 < 0] = 0

        sDMS = self.react_DMS[idx]
        sDMS[sDMS < 0] = 0

        # reactivity = torch.from_numpy(np.stack([self.react_2A3[idx],
        #                                    self.react_DMS[idx]],-1))
        reactivity = torch.from_numpy(np.stack([s2A3, sDMS],-1))
        reactivity_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        # snr = torch.FloatTensor([self.snr_2A3[idx], self.snr_DMS[idx]])
        
        # return {'seq':torch.from_numpy(seq), \
        #         'reactivity': reactivity, \
        #         'reactivity_err': reactivity_err, \
        #         'snr':snr, \
        #         'mask': mask}
        return torch.from_numpy(seq), reactivity, mask
    

# class RNADataLoader:
#     def __init__(self, dataloader, device='cuda:0'):
#         self.dataloader = dataloader
#         self.device = device

#     def __len__(self):
#         return len(self.dataloader)

#     def __iter__(self):
#         for batch in self.dataloader:
#             return tuple({k: x[k].to(self.device) for k in x} for x in batch)