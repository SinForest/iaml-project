# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:36:32 2018

@author: twuensche-uni-hd
"""

import torch
import torch.nn as nn
from dataset import SoundfileDataset

CUDA_ON     = True
DATA_PATH   = "./all_metadata.p"

device = find_device()
dataset = load_data()



def load_data():
    print('=> loading dataset <=')
    dataset = SoundfileDataset(path=DATA_PATH, seg_size=2, hotvec=False, cut_data=True, verbose=False, out_type='sample')
    print('=> dataset loaded <=')
    return dataset

def find_device():
    if (CUDA_ON and not torch.cuda.is_available()):
        raise Exception("No GPU found")

    return torch.device("cuda" if CUDA_ON else "cpu")


def make_model():
    filter_length = 3
    # depth = 9
    padding = (filter_length-1)/2
    modules = []
    
    modules.append(nn.Conv1d(1,128,filter_length,stride=filter_length,padding=padding))
    
    modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(128,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(256,512,filter_length,stride=1,padding=padding))
    modules.append(nn.MaxPool1d(filter_length))
    
    modules.append(nn.Conv1d(512,512,1,stride=1))
    modules.append(nn.Dropout(0.5))
    
    modules.append(nn.Sigmoid())
    
    return nn.Sequential(*modules)