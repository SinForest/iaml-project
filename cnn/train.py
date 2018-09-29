import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from itertools import count

# my imports:
sys.path.append("../")
from dataset import SoundfileDataset
from model import Model

# CONST:
IPATH       = "../dataset.ln"
BATCH_SIZE  = 64
N_PROC      = 2
CUDA_DEVICE = 0 #NOCUDA
N_MELS      = 128

print("### creating dataset ###")
dset = SoundfileDataset("../all_metadata.p", IPATH, seg_size=30, out_type='mel', n_mels=N_MELS)
print("### building model ###")
model = Model(*dset[0][0].shape, dset.n_classes)
if CUDA_DEVICE > -1:
    model.cuda()
crit  = torch.nn.CrossEntropyLoss()
opti  = torch.optim.RMSprop(model.parameters())

print("### starting train loop ###")

for epoch in count(1):

    dloader  = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_PROC, drop_last=True)
    abs_prec = 0
    for X, y in tqdm(dloader, desc=f'epoch {epoch}'):

        with torch.set_grad_enabled(True):

            if CUDA_DEVICE > -1:
                X, y = X.cuda(CUDA_DEVICE), y.cuda(CUDA_DEVICE)
            
            pred = model(X)
            loss = crit(pred, y)
            opti.zero_grad()

            loss.backward()
            opti.step()

        abs_prec += (pred.max(1)[1] == y).sum().item()
    
    prec = abs_prec / len(dloader)
    tqdm.write(f"precision: {prec*100:.2f}%")
