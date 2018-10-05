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

# our homebrew code:
from sample_dcnn_model import Model
#from sample_dcnn_model_fewer_filters import Model
#from sample_dcnn_model_fewer_filters_3_6 import Model

sys.path.append("../")
from dataset import SoundfileDataset

BATCH_SIZE  = 32
N_PROC = 16
num_epochs  = 100
CUDA_ON     = True
SEGMENTS        = 1
FILTER_LENGTH   = 3
DEPTH           = 9
SAMPLES         = (FILTER_LENGTH ** (DEPTH+1)) 
learn_r     = 0.01
import sys



METADATA_PATH   = "../all_metadata.p"
#METADATA_PATH   = "../metadata.ln/tracks.csv"
DATASET_PATH    = "../dataset.ln"

def find_device():
    if (CUDA_ON and not torch.cuda.is_available()):
        raise Exception("No GPU found")

    return torch.device("cuda" if CUDA_ON else "cpu")

    

def main():
    sys.stdout = open('output.txt', 'w')

    device = find_device()
    
    print('=> loading dataset <=')
    dataset = SoundfileDataset(METADATA_PATH, DATASET_PATH, cut_data=False, out_type='sample')
    print('=> dataset loaded <=')
    model = Model(SEGMENTS, SAMPLES, dataset.n_classes)
    model = model.to(device)    
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learn_r, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=0, cooldown=1, verbose=True)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    
    criterion = nn.CrossEntropyLoss()
        
    train_sampler, valid_sampler = dataset.get_split()
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=N_PROC, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=valid_sampler, num_workers=N_PROC, drop_last=True)
        
   
    print('=> begin training <=')
    for epoch in range(0, num_epochs):
        #scheduler.step()
        #current_lr = scheduler.get_lr()
        #tqdm.write(f"learning rate for next epoch: {current_lr:.10f}")
        print(f'training epoch {epoch}')
        # train
        running_loss = 0.0
        abs_prec = 0

        model.train(True)
        
        with torch.set_grad_enabled(True):
            for X, y in tqdm(train_loader, desc=f'training epoch {epoch}'):
                
            
                
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y.long())
                optimizer.zero_grad()
    
                loss.backward()
                optimizer.step()
                running_loss += loss.data
                abs_prec += (pred.max(1)[1] == y).sum().item()
        

        prec = abs_prec / (len(train_loader) * BATCH_SIZE)
        tqdm.write(f"train precision: {prec*100:.2f}%")
        print(f"train precision: {prec*100:.2f}%")
        
        # validate
        running_loss = 0.0
        abs_prec = 0
        model.train(False)
        with torch.set_grad_enabled(False):
            for X, y in tqdm(validation_loader, desc=f'validation epoch {epoch}'):
                
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                loss = criterion(pred, y.long())
                optimizer.zero_grad()
                running_loss += loss.data
                abs_prec += (pred.max(1)[1] == y).sum().item()
        


        prec = abs_prec / (len(validation_loader) * BATCH_SIZE)
        tqdm.write(f"validation precision: {prec*100:.2f}%")
        print(f"validation precision: {prec*100:.2f}%")
        #tqdm.write(f"validation running loss: {running_loss:.4f}%")
        scheduler.step(running_loss)

        torch.save(model, f"./model_E{epoch}_P{int(prec * 1000)}.t")
        
        
if __name__ == '__main__':
    main()