#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:48:41 2018

@author: twuensche-uni-hd
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self,segments, samples, num_labels):
        super().__init__()
        filter_length = 3
        # depth = 9
        padding = int(round((filter_length-1)/2))
        modules = []
        
        modules.append(nn.Conv1d(segments,16,filter_length,stride=filter_length,padding=padding))
        modules.append(nn.BatchNorm1d(16))
        modules.append(nn.ReLU())
        
        modules.append(nn.Conv1d(16,32,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(32))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(32,32,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(32))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(32,64,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(64,64,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(64))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(64,128,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(128,256,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(256,512,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(512,512,1,stride=1))
        modules.append(nn.Dropout(0.5))
        
        #modules.append(nn.Sigmoid())

        self.conv = nn.Sequential(*modules)
        
        self.fc = nn.Sequential(nn.Linear(512, num_labels), nn.Sigmoid())

  

    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(-1, 1, 512)
        x = self.fc(x)
        x = x.mean(1)
        return x
