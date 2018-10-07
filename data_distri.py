import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from hashlib import md5
from dataset import SoundfileDataset

FILTER = False

path='./all_metadata.p'

data = SoundfileDataset(path=path)
train, valid = data.get_indices()
d = pickle.load(open(path, 'rb'))
classes = set()
for i, (key, val) in enumerate(d.items()):
    if FILTER:
        if val['track']['genre_top'] == "": continue
    classes.add(val['track']['genre_top'])
idx2lbl = dict(enumerate(classes))
lbl2idx = {v:k for k,v in idx2lbl.items()}
n_classes = len(classes)
print(n_classes)
absolute = np.zeros(n_classes)
train_occ = np.zeros(n_classes)
val_occ = np.zeros(n_classes)

for i in train:
    train_occ[data.data[i].label] += 1
train_dis = (train_occ / train_occ.sum()) * 100
print(train_occ.sum())

print("Train")
for i, num in enumerate(train_dis):
    print("Genre : " + idx2lbl[i] + ": {:.2f}%".format(num))

for i in valid:
    val_occ[data.data[i].label] += 1
val_dis = (val_occ / val_occ.sum()) * 100
print(val_occ.sum())

print("Val")
for i, num in enumerate(val_dis):
    print("Genre : " + idx2lbl[i] + ": {:.2f}%".format(num))

for i, (k, v) in enumerate(d.items()):
    if FILTER:
        if v['track']['genre_top'] == "": continue
    absolute[lbl2idx[v['track']['genre_top']]] += 1

occu = (absolute / absolute.sum()) * 100
print(len(occu))

plt.figure(figsize=(10, 8))
for i, num in enumerate(occu):
    plt.bar(i, num, color="#"+md5((idx2lbl[i] + "Zysten rocken fett!").encode('utf-8') ).hexdigest()[:6], label=idx2lbl[i])
    plt.text(i-0.5, num+0.2, str("{:.2f}%".format(num)), color='black')
plt.xticks([])
plt.xlabel("genres")
plt.ylabel("percentage")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
plt.title("Distribution of the genres over the unfiltered dataset")
plt.tight_layout()
plt.savefig("./genredist.png")
plt.clf()

plt.figure(figsize=(10, 8))
for i, num in enumerate(train_dis):
    plt.bar(i, num, color="#"+md5((idx2lbl[i] + "Zysten rocken fett!").encode('utf-8') ).hexdigest()[:6], label=idx2lbl[i])
    plt.text(i-0.5, num+0.2, str("{:.2f}%".format(num)), color='black')
plt.xticks([])
plt.xlabel("genres")
plt.ylabel("percentage")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
plt.title("Genre distribution over the training set")
plt.tight_layout()
plt.savefig("./train.png")

plt.clf()
plt.figure(figsize=(10, 8))
for i, num in enumerate(val_dis):
    plt.bar(i, num, color="#"+md5((idx2lbl[i] + "Zysten rocken fett!").encode('utf-8') ).hexdigest()[:6], label=idx2lbl[i])
    plt.text(i-0.5, num+0.2, str("{:.2f}%".format(num)), color='black')
plt.xticks([])
plt.xlabel("genres")
plt.ylabel("percentage")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
plt.title("Genre distribution over the validation set")
plt.tight_layout()
plt.savefig("./val.png")


for i, num in enumerate(occu):
    print("Genre : " + idx2lbl[i] + ": {:.2f}%, {}".format(num, absolute[i]))
print(occu.sum())