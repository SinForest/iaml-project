#!/bin/env python3
import librosa
import librosa.display
import numpy as np
from tqdm import trange
from dataset import SoundfileDataset

fsize = 1024
ssize = 512
num_epochs  = 15
batch_s     = 2
seg_s       = 2
learn_r     = 0.001
log_percent = 0.25
CUDA_ON     = True
SHUFFLE_ON  = False

DATA_PATH   = "./all_metadata.p"

y, sr = librosa.load("/home/flo/IAML/fma_small/099/099214.mp3", duration=30.0, mono=True) #id 4470
dataset = SoundfileDataset(path=DATA_PATH, seg_size=seg_s, hotvec=False, cut_data=True, verbose=False, out_type='entr')

print(dataset.data[6964])

print(dataset.data[6965])

print(y.shape)

def calc_entropy(song):
    fsize = 1024
    ssize = 512
    
    lenY = song.shape[0]
    lenCut = lenY-(lenY%ssize)
    if(lenY < fsize): print("WTF DUDE!")
    
    energy = np.square(song[:lenCut].reshape(-1,ssize))
    energylist = np.concatenate((energy[:-1], energy[1:]), axis=1)

    framelist = energylist.sum(axis=1)
    p = np.nan_to_num(energylist / framelist[:,None]) 
    entropy = -(p * np.nan_to_num(np.log2(p))).sum(axis=1)

    entdif = entropy[:-1] - entropy[1:]
    med = max(entdif.min(), entdif.max(), key=abs)

    blocksize = []
    blocksize.append(1)

    numbox = []
    numbox.append((np.absolute(song[:-1] - song[1:])).sum() + lenY)
    
    uppervalues = (song.reshape(-1, 2)).max(axis=1)
    lowervalues = (song.reshape(-1, 2)).min(axis=1)

    maxScale = int(np.floor(np.log(lenY) / np.log(2)))
    for scale in range(1, maxScale):
        blocksize.append(blocksize[scale-1]*2)

        numcols = len(uppervalues)
        dummy = (uppervalues - lowervalues).sum() + numcols

        rising = np.less(uppervalues[:-1], lowervalues[1:])
        dummy += ((lowervalues[1:] - uppervalues[:-1]) * rising).sum() #sum where signal is rising

        falling = np.greater(lowervalues[:-1], uppervalues[1:])
        dummy += ((lowervalues[:-1] - uppervalues[1:]) * falling).sum() #sum where signal is falling
        
        numbox.append(dummy/blocksize[scale])

        if((numcols % 2) != 0):
            uppervalues = (uppervalues[:-1].reshape(-1, 2)).max(axis=1)
            lowervalues = (lowervalues[:-1].reshape(-1, 2)).min(axis=1)
        else:
            uppervalues = (uppervalues.reshape(-1, 2)).max(axis=1)
            lowervalues = (lowervalues.reshape(-1, 2)).min(axis=1)

    N = np.log(numbox)
    R = np.log(1/np.array(blocksize))
    m = np.linalg.lstsq(R[:,None],N, rcond=None)

    avg = np.average(entropy)
    std = np.std(entropy)
    mxe = np.max(entropy)
    mne = np.min(entropy)
    frd = m[0][0]

    return np.array([avg, std, mxe, mne, med, frd])

vec = calc_entropy(y)

print(vec)