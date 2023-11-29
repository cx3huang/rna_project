import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split, KFold
import random

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
        

        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if 'reactivity_error_0' in c]].values
        
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
        s2A3[np.isnan(s2A3)] = 0

        sDMS = self.react_DMS[idx]
        sDMS[sDMS < 0] = 0
        sDMS[np.isnan(sDMS)] = 0

        reactivity = torch.from_numpy(np.stack([s2A3, sDMS],-1))
        reactivity_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        # snr = torch.FloatTensor([self.snr_2A3[idx], self.snr_DMS[idx]])

        return torch.from_numpy(seq), reactivity, mask


class LengthMatchingBatchSampler(BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for i in self.sampler:
            s = self.sampler.data_source[i]
            # print(s)
            L = s[2].sum()
            L = max(1, L // 16) # embedding dimension

            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(i)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch

                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [i for b in buckets for i in b]

        for i in leftover:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch

                yielded += 1
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch
            yielded += 1