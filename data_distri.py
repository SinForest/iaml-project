import pickle
import matplotlib as plt
import numpy as np
from tqdm import tqdm

FILTER = True

path='./all_metadata.p'
d = pickle.load(open(path, 'rb'))
classes = set()
for key, val in d.items():
    classes.add(val['track']['genre_top'])
idx2lbl = dict(enumerate(classes))
lbl2idx = {v:k for k,v in idx2lbl.items()}
n_classes = len(classes)
print(n_classes)
occu = np.zeros(n_classes)

for k, v in d.items():
    occu[lbl2idx[v['track']['genre_top']]] += 1

if FILTER:
    occu = occu[1:]
occu = (occu / occu.sum()) * 100

for i, num in enumerate(occu):
    print("Genre : " + idx2lbl[i+FILTER] + ": {:.2f}%".format(num))
print(occu.sum())