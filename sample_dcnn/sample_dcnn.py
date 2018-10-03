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

BATCH_SIZE  = 10
N_PROC = 0
num_epochs  = 1
CUDA_ON     = False
SEGMENTS        = 10
FILTER_LENGTH   = 3
DEPTH           = 9
SAMPLES         = (FILTER_LENGTH ** (DEPTH+1))    

#METADATA_PATH   = "../all_metadata.p"
METADATA_PATH   = "../metadata.ln/tracks.csv"
DATASET_PATH    = "../dataset.ln"

def find_device():
    if (CUDA_ON and not torch.cuda.is_available()):
        raise Exception("No GPU found")

    return torch.device("cuda" if CUDA_ON else "cpu")




def main():

    device = find_device()
    
    print('=> loading dataset <=')
    dataset = SoundfileDataset(METADATA_PATH, DATASET_PATH, cut_data=True, out_type='sample')
    print('=> dataset loaded <=')
    model = Model(SEGMENTS, SAMPLES, dataset.n_classes)
    model = model.to(device)    

    
    optimizer = optim.RMSprop(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    print('=> begin training <=')
    for epoch in range(0, num_epochs):
    
        dataloader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_PROC, drop_last=True)
        abs_prec = 0
        
        
        for X, y in tqdm(dataloader, desc=f'epoch {epoch}'):

            with torch.set_grad_enabled(True):
                
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y.long())
                optimizer.zero_grad()
    
                loss.backward()
                optimizer.step()
    
            abs_prec += (pred.max(1)[1] == y).sum().item()
        
        prec = abs_prec / len(dataloader)
        tqdm.write(f"precision: {prec*100:.2f}%")
        
if __name__ == '__main__':
    main()