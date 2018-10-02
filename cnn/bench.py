import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from chrono import Timer
import time
from itertools import repeat

# my imports:
sys.path.append("../")
from dataset import SoundfileDataset
from model import Model

# CONST:
IPATH       = "../dataset.ln"
BATCH_SIZE  = 64
N_PROC      = 32
CUDA_DEVICE = 0 #NOCUDA
N_MELS      = 128
N_RUNS      = 100

print("### creating dataset ###")
dset = SoundfileDataset("../all_metadata.p", IPATH, seg_size=30, out_type='mel', n_mels=N_MELS, random_slice=N_RUNS*BATCH_SIZE+1)#, cut_data=True)
print("### building model ###")
model = Model(*dset[0][0].shape, dset.n_classes)
if CUDA_DEVICE > -1:
    model.cuda()
crit  = torch.nn.CrossEntropyLoss()
opti  = torch.optim.RMSprop(model.parameters())

print("### datagen. w/o training ###")

dloader  = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_PROC, drop_last=True)
abs_prec = 0

times = []
old = time.perf_counter()
for X, y in dloader:
    now = time.perf_counter()
    times.append(now - old)
    print(f"{times[-1]:.2f}")
    old = time.perf_counter()

X_old, y_old = X, y

print("### training w/o data ###")
abs_prec = 0
for X, y in repeat((X_old, y_old), N_RUNS):
    with Timer() as timer:
        with torch.set_grad_enabled(True):
            if CUDA_DEVICE > -1:
                X, y = X.cuda(CUDA_DEVICE), y.cuda(CUDA_DEVICE)
            
            pred = model(X)
            loss = crit(pred, y)
            opti.zero_grad()

            loss.backward()
            opti.step()

        abs_prec += (pred.max(1)[1] == y).sum().item()
    times.append(timer.elapsed)
    print(f"times[-1]:.2f")

