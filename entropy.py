#!/bin/env python3
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

fsize = 1024
ssize = 512

y, sr = librosa.load("/home/flo/IAML/fma_small/000/000002.mp3", mono=True)
print(y.shape)

framelist = []
for i in range(0, (y.shape[0]-fsize), ssize):
    energy = 0
    for x in range(fsize):
        energy += y[i+x]**2
    framelist.append(energy)
    
entropy = []
for n, i in enumerate(range(0, (y.shape[0]-fsize), ssize)):
    entr = 0
    for x in range(fsize):
        if(y[i+x] != 0):
            p = y[i+x]**2 / framelist[n]
            entr += p * np.log2(p)
    entropy.append(-entr)

med = 0
for i in range(len(entropy)-1):
    dif = entropy[i] - entropy[i+1]
    if(abs(dif) > abs(med)):
        med = dif

smax = int(y.shape[0] / 2)
N = []
R = []
for s in trange(2, smax):
    lenY = y.shape[0]
    r = s / lenY
    nra = 0
    frac = y[:lenY-(lenY%s)].reshape(-1,s)
    subMax = np.max(frac, axis=1)
    subMin = np.min(frac, axis=1)
    nr = ((subMax - subMin) + 1)/s
    Nr = np.sum(np.ceil(nr))
    N.append(Nr)
    R.append(r)

N = np.log(N)
R = np.log(1/np.array(R))
m = np.linalg.lstsq(R[:,None],N)

entropy = np.array(entropy)

avg = np.average(entropy)
std = np.std(entropy)
mxe = np.max(entropy)
mne = np.min(entropy)
frd = m[0]

print(avg, std, mxe, mne, med, frd)
print("Dim: {}".format(m))