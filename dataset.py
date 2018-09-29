#!/bin/env python3
# requires python3.6 or higher (Hooray for f-Strings! \o/)
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

    def __init__(self, path, ipath="./dataset.ln", seg_size=30, hotvec=False, cut_data=False, verbose=True, out_type='raw', n_mels=128):
        _, ext = os.path.splitext(path)
        if ext == ".p":
            d = pickle.load(open(path, 'rb'))
        elif ext in ['.csv', '.txt']:
            d = read_dict(path)
        else:
            raise RuntimeError(f"{path}: extention '{ext[1:]}' not known")
        
        # Remove non-existent data points (e.g. b/c of smaller subset)
        tmp_len = len(d)
        d = {k:v for k,v in d.items() if os.path.isfile(os.path.join(ipath, v['path'])) and int(v["track"]["duration"]) > seg_size}
        if verbose:
            print(f"removed {tmp_len - len(d)}/{tmp_len} non-existing/too short items" )

        # Generate class-idx-converter
        classes = set()
        for key, val in tqdm(d.items(), desc="build class set"):
            classes.add(val['track']['genre_top'])
        self.idx2lbl = dict(enumerate(classes))
        self.lbl2idx = {v:k for k,v in self.idx2lbl.items()}
        self.n_classes = len(classes)

        # Copy neccecary data into list of named tuples for quick access
        Struct = namedtuple("Data", "id path duration label")
        self.data = []
        for key, val in tqdm(d.items(), desc="build dataset"):
            try:          # |id is actually not needed here
                tmp = Struct(id=key, path=val['path'], duration=int(val["track"]["duration"]),
                             label=self.lbl2idx[ val['track']['genre_top'] ])
                             #labels=[int(x) for x in val['track']['genres_all'][1:-1].split(",")])
            except ValueError as e:
                continue # I forgot why... maybe some tracks have no labels; damn, comment immediately, me!
            
            self.data.append(tmp)

        self.seg_size = seg_size # size of random crops
        self.ipath = ipath       # path of image data
        self.hotvec = hotvec     # whether to return labels as one-hot-vec
        self.cut_data = cut_data # whether data is only 30s per song
        self.out_type = out_type # 'raw' or 'mel'
        self.n_mels = n_mels
    
    def calc_mel(self, song, sr):
        n_fft = 2**11         # shortest human-disting. sound (music)
        hop_length = 2**10    # => 50% overlap of frames

        return librosa.feature.melspectrogram(song, sr=sr, n_mels=self.n_mels, n_fft=n_fft, hop_length=hop_length)


    def __getitem__(self, idx):
        #TODO: benchmark by iterating over pass'ing Dataloader

        this = self.data[idx]

        offs = np.random.randint((this.duration if not self.cut_data else 31) - self.seg_size) # offset to start random crop
        try:
            song, sr = librosa.load(os.path.join(self.ipath, this.path), mono=True, offset=offs, duration=self.seg_size)
            # (change resampling method, if to slow)

            if self.out_type == 'raw':
                X = song
            elif self.out_type == 'mel':
                X = self.calc_mel(song, sr)
            else:
                raise ValueError(f"wrong out_type '{self.out_type}'")
    # do we really need to log(S) this? skip this for first attempts
        except Exception as e:
            print(f"offs:{offs}; dur:{this.duration}; len:{len(song)}; pth:{this.path}")
            raise e
        del song, sr

        # create hot-vector (if needed)
        if self.hotvec:
            y = torch.zeros(self.n_classes)
            y[this.label] = 1
        else:
            y = this.label

        return torch.as_tensor(X, dtype=torch.float32), y
        
    def __len__(self):
        return len(self.data)
    

if __name__ == "__main__":

    IPATH = "./dataset.ln"  # => README.MD

    dset = SoundfileDataset("./all_metadata.p", IPATH, seg_size=30, cut_data=True, out_type='mel')
    
    print(len(dset))
    X, y = dset[0]
    print(X.shape, X.mean(), y)
    print(dset.idx2lbl[y])
    dset.hotvec = True
    X, y = dset[1]
    print(X.shape, X.mean(), y)

    print("### Benchmarking dataloading speed ###")
    #TODO: compare to training with offline-preprocessed data, to see if preprocessing is bottleneck
    dataloader = DataLoader(dset, num_workers=32, batch_size=64)
    for X, y in tqdm(dataloader):
        pass
