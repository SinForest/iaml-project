#!/usr/bin/env python3.7
import numpy as np
import torch
import pickle
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
from torch import Tensor as Ten
from dataset import SoundfileDataset
from entrset import EntropyDataset
import operator
import sys, os

num_epochs  = 100
batch_s     = 2**12
seg_s       = 2
learn_r     = 0.01
s_factor    = 0.5
log_percent = 0.25
CUDA_ON     = True
SHUFFLE_ON  = True

DATA_PATH   = "./all_metadata.p"
MODEL_PATH  = "../models/"

dataset = EntropyDataset()
trainsamp, valsamp = dataset.getsplit()
trainloader = torch.utils.data.DataLoader(trainsamp, batch_size=batch_s, shuffle=SHUFFLE_ON, num_workers=4)
valloader   = torch.utils.data.DataLoader(valsamp, batch_size=batch_s, shuffle=False, num_workers=4)
log_interval = np.ceil((len(trainloader.dataset) * log_percent) / batch_s)

""" n_con = 1024
model = nn.Sequential(
    nn.Linear(6, n_con),
    nn.PReLU(num_parameters=n_con),
    nn.BatchNorm1d(n_con),
    nn.Dropout(p=0.5),
    nn.Linear(n_con, n_con),
    nn.PReLU(num_parameters=n_con),
    nn.BatchNorm1d(n_con),
    nn.Dropout(p=0.5),
    nn.Linear(n_con, dataset.n_classes)
) """

l1 = 1617
l2 = 3036
p = 0.4907947467956333
pre = 1
model = nn.Sequential(
    nn.Linear(6, l1),
    nn.Tanh(),
    nn.BatchNorm1d(l1),
    #nn.Dropout(p=p),
    nn.Linear(l1, l2),
    nn.Tanh(),
    nn.BatchNorm1d(l2),
    #nn.Dropout(p=p),
    nn.Linear(l2, dataset.n_classes)
)


for m in model.modules():
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        m.running_mean.zero_()
        m.running_var.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

optimizer = optim.Adam(model.parameters(), lr=learn_r)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=s_factor, patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

model.cuda()

def train(epoch):
    total_loss = 0
    total_size = 0
    model.train()
    accuracy = 0
    for batch_id, (data, target) in enumerate(trainloader):
        
        data = data.type(torch.FloatTensor)

        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)

        total_loss += loss.item()
        total_size += data.size(0)

        accur = (torch.argmax(output, dim=1) == target).sum()
        accuracy += accur 

        loss.backward()
        optimizer.step()
        
        if batch_id % log_interval == 0:
            print('Train Epoch: {} [{:>5d}/{:> 5d} ({:>2.0f}%)]\tCurrent loss: {:.6f}\t Current accuracy: {:.2f}'.format(
            epoch, total_size, len(trainloader.dataset), 100. * batch_id / len(trainloader), loss.item() / data.size(0), float(accur) / float(batch_s)))

    total_acc = float(accuracy) / float(total_size)
    print('Train Epoch: {}, Entropy average loss: {:.6f}, Entropy average accuracy: {:.4f}'.format(
            epoch, total_loss / total_size, total_acc))
    
    return (total_loss/total_size , total_acc)

def validate():
    model.eval()
    val_loss = 0
    accur = 0
    for data, target in valloader:
        
        if CUDA_ON:
            data, target = data.cuda(), target.cuda()
        
        output = model(data)
        
        val_loss += criterion(output, target).item()
        accur += (torch.argmax(output, dim=1) == target).sum()
    
    val_loss /= len(valloader.dataset)
    accuracy = float(accur) / float(len(valloader.dataset))
    
    print('\nTest set: Entropy average loss: {:.6f}, average accuracy: {:.4f}'.format(val_loss, accuracy))
    
    return val_loss, accuracy

for epoch in range(0, num_epochs):
    loss, acc = train(epoch)
    val_l, val_a = validate()
    scheduler.step(val_l)

def acc_per_class():
    model.eval()
    occ = np.zeros(16)
    hit = np.zeros(16)
    for x, y in valloader:

        x, y = x.cuda(), y.cuda()

        output = model(x)
        comp = (torch.argmax(output, dim=1) == y)
        
        for i, lbl in enumerate(y):
            occ[lbl] += 1
            hit[lbl] += comp[i]
    
    acc = (hit / occ) * 100

    return acc, hit, occ

acc, hit, occ = acc_per_class()

for i, elem in enumerate(acc):
    print("{}: Accuracy {}%, {} from {}".format(i, elem, hit[i], occ[i]))
print((hit.sum()/occ.sum())*100)