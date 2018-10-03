import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from itertools import count

# my imports:
sys.path.append("../")
from dataset import SoundfileDataset
from model import Model

# ANSI Esc:
_CR = "\033[31m"
_CG = "\033[32m"
_CB = "\033[36;1m"

_XX = "\033[0m"

# CONST:
IPATH        = "../melsset.ln"
BATCH_SIZE   = 2
N_PROC       = 32
CUDA_DEVICE  = -1 #NOCUDA
MEL_SEG_SIZE = 128
LOG_COUNT    = 20

print("### creating dataset ###")
dset = SoundfileDataset("../all_metadata.p", IPATH, seg_size=30, out_type='pre_mel', mel_seg_size=MEL_SEG_SIZE, verbose=True)
print("### splitting dataset ###")
tset, vset = dset.get_split(sampler=False)
print("### initializing dataloader ###")
tloader = DataLoader(tset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_PROC, drop_last=True)
vloader = DataLoader(vset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_PROC, drop_last=True)

print("### building model ###")
model = Model(*dset[0][0].shape, dset.n_classes)
if CUDA_DEVICE > -1:
    model.cuda()
crit  = torch.nn.CrossEntropyLoss()
opti  = torch.optim.RMSprop(model.parameters())

print("### starting train loop ###")

for epoch in count(1):

    abs_prec = 0
    losses = []
    model.train()
    for i, (X, y) in enumerate(tqdm(tloader, desc=f'epoch {epoch}'), 1):

        with torch.set_grad_enabled(True):

            if CUDA_DEVICE > -1:
                X, y = X.cuda(CUDA_DEVICE), y.cuda(CUDA_DEVICE)
            
            pred = model(X)
            loss = crit(pred, y)
            opti.zero_grad()

            loss.backward()
            opti.step()

        losses.append(loss.item())
        abs_prec += (pred.max(1)[1] == y).sum().item()

        if i % LOG_COUNT == 0:
            tqdm.write(f"{_CB}B[{i:>4}/{len(tloader)}]{_XX}: r.prec: {(abs_prec * 100) / (len(X)*i):>2.2f}; loss: {sum(losses)/len(losses):>2.3f}")
        
        
    
    prec = abs_prec / (len(X) * len(tloader))
    tqdm.write(f"{_CR}EPOCH[{epoch}]{_XX}: r.prec: {prec * 100:>2.2f}; loss: {sum(losses)/len(losses):>2.3f}")

    abs_prec = 0
    model.eval()
    for i, (X, y) in enumerate(tqdm(vloader, desc=f'valid {epoch}'), 1):
        
        with torch.set_grad_enabled(False):
            if CUDA_DEVICE > -1:
                X, y = X.cuda(CUDA_DEVICE), y.cuda(CUDA_DEVICE)
            
            pred = model(X)
        abs_prec += (pred.max(1)[1] == y).sum().item()
    
    tqdm.write(f"{_CG}VALID[{epoch}]{_XX}: r.prec: {prec * 100:>2.2f}; loss: {sum(losses)/len(losses):>2.3f}")

