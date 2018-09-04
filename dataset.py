import torch
from torch.utils.data import Dataset, DataLoader
from read_csv import read_dict
import os
from collections import namedtuple
from random import randint
import pickle
import pydub
from scipy.io import wavfile
import tempfile
from tqdm import tqdm
import numpy as np

class SoundfileDataset(Dataset):

    def __init__(self, path, ipath, output='raw', seg_size=30_000):
        _, ext = os.path.splitext(path)
        if ext == ".p":
            d = pickle.load(open(path, 'rb'))
        elif ext in ['.csv', '.txt']:
            d = read_dict(path)
        else:
            raise RuntimeError(f"{path}: extention '{ext[1:]}' not known")
        
        self.data = []
        self.classes = []
        Struct = namedtuple("Data", "id path label labels")

        for key, val in tqdm(d.items()):
            try:
                tmp = Struct(id=key, path=val['path'],
                         label=[int(x) for x in val['track']['genres'][1:-1].split(",")],
                         labels=[int(x) for x in val['track']['genres_all'][1:-1].split(",")])
            except ValueError as e:
                continue
            # if len(tmp.label) != 1:
            #     print(RuntimeWarning(f"multiple labels for '{tmp.id}': {tmp.label}"))
            for lbl in tmp.label:
                if lbl not in self.classes:
                    self.classes.append(tmp.label[0])
            
            self.data.append(tmp)
        
        self.seg_size = seg_size
        self.ipath = ipath

    def __getitem__(self, idx):
        this = self.data[idx]
        seg = pydub.AudioSegment.from_mp3(os.path.join(self.ipath, this.path))
        sli = randint(0, len(seg) - self.seg_size - 1) # random slice, off by one error?
        seg = seg[sli: sli + self.seg_size]
        f = tempfile.NamedTemporaryFile()
        seg.export(f.name, format='wav')
        rate, data = wavfile.read(f.name)
        f.close()

        if self.output == 'raw':
            return torch.Tensor(data), this.label
        elif self.output == 'fft':
            return torch.Tensor(np.fft.fft(data)), this.label
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":

    IPATH = "/home/cobalt/datasets/fma/fma_small/"

    dset = SoundfileDataset("./all_metadata.p", IPATH, seg_size=10_000)
    print(len(dset))
    print(dset[0])