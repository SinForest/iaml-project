import torch
from torch.utils.data import Dataset, DataLoader
from read_csv import read_dict
import os
from collections import namedtuple
import pickle
from tqdm import tqdm
import numpy as np
import librosa

class SoundfileDataset(Dataset):

    def __init__(self, path, ipath, output='raw', seg_size=30):
        _, ext = os.path.splitext(path)
        if ext == ".p":
            d = pickle.load(open(path, 'rb'))
        elif ext in ['.csv', '.txt']:
            d = read_dict(path)
        else:
            raise RuntimeError(f"{path}: extention '{ext[1:]}' not known")
        
        self.data = []
        classes = []
        Struct = namedtuple("Data", "id path duration label labels")

        for key, val in tqdm(d.items(), desc="build dataset"):
            if not os.path.isfile(os.path.join(ipath, val['path'])):
                continue  # skip, if the file is not present
            try:
                tmp = Struct(id=key, path=val['path'], duration=val["track"]["duration"],
                             label=[int(x) for x in val['track']['genres'][1:-1].split(",")],
                             labels=[int(x) for x in val['track']['genres_all'][1:-1].split(",")])
            except ValueError as e:
                continue # I forgot why... maybe some tracks have no labels

            for lbl in tmp.label:
                if lbl not in classes:
                    classes.append(lbl)
            
            self.data.append(tmp)
        
        # Generate class-idx-converter
        self.idx2lbl = dict(enumerate(classes))
        self.lbl2idx = {v:k for k,v in self.idx2lbl.items()}
        self.n_classes = len(classes)

        self.seg_size = seg_size
        self.ipath = ipath

    def __getitem__(self, idx):
        n_fft = 2*12
        hop_length = 2*11
        n_mels = 64
        #TODO: make this(^) to parameters and/or find good values

        this = self.data[idx]

        offs = np.random.randint(this.duration) # offset to start random crop
        song, sr = librosa.load(os.path.join(self.ipath, this.path), mono=True, offset=offs, duration=self.seg_size)
        # (change resampling method, if to slow)

        S = librosa.feature.melspectrogram(song, sr=sr, n_mels=64, n_fft=n_fft, hop_length=hop_length)
        # do we really need to log(S) this? skip this for first attempts

        #TODO: (maybe create more fetures?)
        #TODO: create hot-vector
        #TODO: return everything
        
    
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":

    IPATH = "/home/cobalt/datasets/fma/fma_small/"

    dset = SoundfileDataset("./all_metadata.p", IPATH, seg_size=10)
    print(len(dset))
    print(dset[0])