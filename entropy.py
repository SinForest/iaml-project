#!/usr/bin/env python3
import numpy as np
import torch
import pickle
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as Var
from torch import Tensor as Ten
from dataset import SoundfileDataset
import operator
import sys, os

num_epochs  = 15
batch_s     = 2
seg_s       = 2
learn_r     = 0.001
log_percent = 0.01
CUDA_ON     = True
SHUFFLE_ON  = False

DATA_PATH   = "./all_metadata.p"
MODEL_PATH  = "../models/"

dataset = SoundfileDataset(path=DATA_PATH, seg_size=seg_s, hotvec=False, cut_data=True, verbose=False, out_type='entr')
trainsamp, valsamp = dataset.get_split(sampler=False)
trainloader = torch.utils.data.DataLoader(trainsamp, batch_size=batch_s, shuffle=SHUFFLE_ON, num_workers=6)
valloader   = torch.utils.data.DataLoader(valsamp, batch_size=batch_s, shuffle=SHUFFLE_ON, num_workers=6)
log_interval = np.ceil((len(trainloader.dataset) * log_percent) / batch_s)

print(dataset.n_classes)

model = nn.Sequential(
            nn.Linear(6, 2048),
            nn.PReLU(num_parameters=2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.PReLU(num_parameters=2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.PReLU(num_parameters=2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=0.5),
            nn.Linear(2048, dataset.n_classes)
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
criterion = nn.CrossEntropyLoss()

if CUDA_ON:
    model.cuda()

print(sum([sum([y.numel() for y in x.parameters()]) for x in model.modules() if type(x) not in {nn.Sequential}]))

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
    print('Train Epoch: {}, Entropy average loss: {:.6f}, Entropy average accuracy: {:.2f}'.format(
            epoch, total_loss / total_size, total_acc))
    
    return (total_loss/total_size , total_acc)

loss_list = []
acc_list  = []
print("Beginning Training")
for epoch in range(0, num_epochs):
    loss, acc = train(epoch)
    loss_list.append(loss)
    acc_list.append(acc)
    state = {'state_dict':model.state_dict(), 'optim':optimizer.state_dict(), 'epoch':epoch, 'train_loss':loss_list, 'accuracy': acc_list}
    filename = "../models/entropy_{:02d}.nn".format(epoch)
    torch.save(state, filename)