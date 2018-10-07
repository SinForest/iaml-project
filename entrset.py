import torch
from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np

class EntropyDataset(Dataset):

    def __init__(self, path="./entrset.ln", FILTER=True):
        self.ent = np.load(os.path.join(path, 'entropy.npy'))
        print(self.ent.shape)
        self.lbl = np.load(os.path.join(path, 'labels.npy'))
        print(self.lbl.shape)

        if FILTER:
            self.ent = np.array(self.ent[self.lbl != 0])
            self.lbl = np.array(self.lbl[self.lbl != 0])
            self.lbl = self.lbl - 1
            print(self.ent.shape)
            print(self.lbl.shape)

        classes = set()
        for x in self.lbl:
            classes.add(x)
        self.n_classes = len(classes)
        print(self.n_classes)

    def __getitem__(self, idx):
        X = self.ent[idx]
        y = self.lbl[idx]

        return torch.as_tensor(X, dtype=torch.float32), y

    def __len__(self):
        return self.ent.shape[0]

    def getsplit(self, val_size=0.2, shuffle=True):
        seed = 4 #such evil
        datasize = self.__len__()
        indices = list(range(datasize))
        split = int(np.floor(val_size * datasize))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_set = Subset(self, train_indices)
        valid_set = Subset(self, val_indices)

        return train_set, valid_set

    def getindices(self, val_size=0.2, shuffle=True):
        seed = 4 #such evil
        datasize = self.__len__()
        indices = list(range(datasize))
        split = int(np.floor(val_size * datasize))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        return train_indices, val_indices