import sys

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# my imports:
sys.path.append("../")
from dataset import SoundfileDataset
from model import Model

# CONST:
IPATH       = "../dataset.ln"
BATCH_SIZE  = 4
N_PROC      = 2
CUDA_DEVICE = -1 #NOCUDA
N_MELS      = 128

print("### creating dataset ###")
dset = SoundfileDataset("../all_metadata.p", IPATH, seg_size=30, cut_data=True, out_type='mel', n_mels=N_MELS)
print("### building model ###")
model = Model(*dset[0][0].shape(), dset.n_classes)
if CUDA_DEVICE > -1:
    model.cuda()
crit  = torch.nn.CrossEntropyLoss()
opti  = torch.nn.opti.RMSprop(model.parameters())

print("### starting train loop ###")
for X, y in tqdm(DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_PROC, drop_last=True)):

    with torch.set_grad_enabled(True):

        if CUDA_DEVICE > -1:
            X, y = X.cuda(CUDA_DEVICE), y.cuda(CUDA_DEVICE)
        
        pred = model(X)
        loss = crit(pred, y)
        opti.zero_grad()

        loss.backward()
        opti.step()