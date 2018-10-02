#!/bin/env python3
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

fsize = 1024
ssize = 512

y, sr = librosa.load("/home/flo/IAML/fma_small/000/000140.mp3", duration=30.0, mono=True)
lenY = y.shape[0]
lenCut = lenY-(lenY%ssize)
print(lenY, lenCut)

energy = (y[:lenCut].reshape(-1,ssize))**2
energylist = np.concatenate((energy[:-1], energy[1:]), axis=1)

framelist = energylist.sum(axis=1)

p = energylist / framelist[:,None]
entropy = -(p * np.nan_to_num(np.log2(p))).sum(axis=1)

entdif = entropy[:-1] - entropy[1:]
med = max(entdif.min(), entdif.max(), key=abs)

BlockSize = []
BlockSize.append(1)
NumBox = []
NumBox.append(1)

for i in range(0, lenY-1):
    NumBox[0] += 1 + abs(y[i]-y[i+1])

NumCols = int(lenY / 2)
UpperValue = [None] * NumCols
LowerValue = [None] * NumCols

for i in range(0, NumCols):
    UpperValue[i] = max(y[2*i], y[2*i+1])
    LowerValue[i] = min(y[2*i], y[2*i+1])

maxScale = int(np.floor(np.log(lenY) / np.log(2)))

for scale in range(1, maxScale):
    dummy = 0
    BlockSize.append(BlockSize[scale-1]*2)
    for i in range(0, NumCols-1):
        dummy += UpperValue[i] - LowerValue[i] + 1
        if UpperValue[i] < LowerValue[i+1]:
            dummy += LowerValue[i+1] - UpperValue[i]
        if LowerValue[i] > UpperValue[i+1]:
            dummy += LowerValue[i] - UpperValue[i+1]
    
    NumBox.append(dummy/BlockSize[scale])
    NumCols = int(NumCols / 2)
    for i in range (0, NumCols):
        UpperValue[i] = max(UpperValue[2*i], UpperValue[2*i+1])
        LowerValue[i] = min(LowerValue[2*i], LowerValue[2*i+1])

BlockSize = np.array(BlockSize)
NumBox = np.array(NumBox)
print("SHAPES")
print(BlockSize)
print(NumBox)

N = np.log(NumBox)
R = np.log(1/BlockSize)
m = np.linalg.lstsq(R[:,None],N)
print(m[0][0])


""" smax = int(y.shape[0] / 2)
N = []
R = []
for s in trange(2, smax):
    cutS = lenY - (lenY%s)
    r = s / lenY
    frac = y[:cutS].reshape(-1,s)
    subMax = np.max(frac, axis=1)
    subMin = np.min(frac, axis=1)
    nr = ((subMax - subMin) + 1)/s
    Nr = np.sum(np.ceil(nr))
    N.append(Nr)
    R.append(r)

N = np.log(N)
R = np.log(1/np.array(R))
m = np.linalg.lstsq(R[:,None],N) """

avg = np.average(entropy)
std = np.std(entropy)
mxe = np.max(entropy)
mne = np.min(entropy)
frd = m[0][0]

print(avg, std, mxe, mne, med, frd)