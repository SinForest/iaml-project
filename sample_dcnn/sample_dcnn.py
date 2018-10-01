# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:36:32 2018

@author: twuensche-uni-hd
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# our homebrew code:
from sample_dcnn_model import Model
sys.path.append("../")
from dataset import SoundfileDataset

BATCH_SIZE  = 64
N_PROC = 32
num_epochs  = 1
CUDA_ON     = True

DATA_PATH   = "../metadata.ln/tracks.csv"

def find_device():
    if (CUDA_ON and not torch.cuda.is_available()):
        raise Exception("No GPU found")

    return torch.device("cuda" if CUDA_ON else "cpu")




def main():
    device = find_device()
    
    print('=> loading dataset <=')
    dataset = SoundfileDataset(path=DATA_PATH, out_type='sample')
    print('=> dataset loaded <=')
    
    model = Model(dataset.n_classes).to(device)
    
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(0, num_epochs):
    
        dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_PROC, drop_last=True)
        abs_prec = 0
        for X, y in tqdm(dataloader, desc=f'epoch {epoch}'):
    
            with torch.set_grad_enabled(True):
    
                pred = model(X)
                loss = criterion(pred, y)
                optimizer.zero_grad()
    
                loss.backward()
                optimizer.step()
    
            abs_prec += (pred.max(1)[1] == y).sum().item()
        
        prec = abs_prec / len(dataloader)
        tqdm.write(f"precision: {prec*100:.2f}%")
        
if __name__ == '__main__':
    main()