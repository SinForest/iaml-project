#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 16:11:08 2018

@author: twuensche
"""


import torch.nn as nn


class Model(nn.Module):
    def __init__(self,segments, samples, num_labels):
        super().__init__()
        filter_length = 2
        # depth = 14
        padding = 1
        modules = []
        
        modules.append(nn.Conv1d(segments,128,filter_length,stride=filter_length,padding=padding))
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
        
        modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
        modules.append(nn.Conv1d(128,128,filter_length,stride=1,padding=padding))
        modules.append(nn.BatchNorm1d(128))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(filter_length))
        
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
        modules.append(nn.Dropout(0.5))
        

        self.conv = nn.Sequential(*modules)
        
        self.fc = nn.Sequential(nn.Linear(512, num_labels), nn.Sigmoid())

  

    def forward(self, x):
        
        x = self.conv(x)
        x = x.view(-1, 1, 512)
        x = self.fc(x)
        x = x.mean(1)
        return x