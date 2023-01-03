from random import randint
import numpy as np
import torch
from torch.utils.data import Dataset

class trajectory_dataset(Dataset):
    def __init__(self, fnames, seqlen=100, tgtlen=10, device="cuda"):
        data = []
        for fname in fnames:
            csv = np.loadtxt(fname, delimiter=",", dtype=np.float32)
            #csv = csv.reshape((1,*data.shape))
            data.append(csv)
        self.data = torch.from_numpy(np.stack(data,axis=0)).to(device)
        self.seqlen = seqlen
        self.tgtlen = tgtlen
        self.trials = len(fnames)

    def __len__(self):
        return self.data.size()[0]*(self.data.size()[1]-self.seqlen-self.tgtlen)

    def __getitem__(self, idx):
        idx_trial = idx // (self.data.size()[1]-self.seqlen-self.tgtlen)
        if idx_trial > self.trials - 1:
            idx_trials = randint(0,self.trials-1)
        idx_sample = idx % (self.data.size()[1]-self.seqlen-self.tgtlen)
        x = self.data[idx_trial,idx_sample:idx_sample+self.seqlen,:]
        y = self.data[idx_trial,idx_sample+self.seqlen:idx_sample+self.seqlen+self.tgtlen,:]
        return x,y