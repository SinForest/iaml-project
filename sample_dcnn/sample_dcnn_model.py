# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 16:35:45 2018

@author: twuensche-uni-hd
"""

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        filter_length = 3
        # depth = 9
        padding = (filter_length-1)/2
        modules = []
        
        modules.append(nn.Conv1d(1,128,filter_length,stride=filter_length,padding=padding))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        
        modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
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
        
        modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(256))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(256,256,filter_length,stride=1,padding=padding))
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
        modules.append(nn.BatchNorm1d(512))
        modules.append(nn.ReLU())
        modules.append(nn.Dropout(0.5))
        
        #modules.append(nn.Sigmoid())

        self.conv = nn.Sequential(*modules)
        
        self.fc = nn.Sequential(nn.Linear(512, num_labels))

  

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
