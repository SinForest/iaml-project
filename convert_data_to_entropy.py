import numpy as np
import librosa
import os
import pickle
import multiprocessing
from tqdm import tqdm

INPATH  = "./dataset.ln"
OUTPATH = "./entrset.ln"
np.seterr(all='ignore')

def calc_entropy(path):

    song, sr = librosa.load(path, mono=True, duration=120)
    fsize = 1024
    ssize = 512
    
    lenY = song.shape[0]
    lenCut = lenY - (lenY % ssize)
    if(lenY < fsize):
        print("SONG TOO SHORT!!!")
        return np.array([0, 0, 0, 0, 0, 0])

    energy = np.square(song[:lenCut].reshape(-1,ssize))
    energylist = np.concatenate((energy[:-1], energy[1:]), axis=1)

    framelist = energylist.sum(axis=1)
    p = np.nan_to_num(energylist / framelist[:,None]) #whole frame might be 0 causing division by zero
    entropy = -(p * np.nan_to_num(np.log2(p))).sum(axis=1) #same goes for log

    blocksize = []
    blocksize.append(1)

    numbox = []
    numbox.append((np.absolute(song[:-1] - song[1:])).sum() + lenY)

    if((lenY % 2) != 0): #double boxsize, half max and min
        uppervalues = (song[:-1].reshape(-1, 2)).max(axis=1)
        lowervalues = (song[:-1].reshape(-1, 2)).min(axis=1)
    else:
        uppervalues = (song.reshape(-1, 2)).max(axis=1)
        lowervalues = (song.reshape(-1, 2)).min(axis=1)

    maxScale = int(np.floor(np.log(lenY) / np.log(2)))
    for scale in range(1, maxScale):
        blocksize.append(blocksize[scale-1]*2)

        numcols = len(uppervalues)
        dummy = (uppervalues - lowervalues).sum() + numcols

        rising = np.less(uppervalues[:-1], lowervalues[1:])
        dummy += ((lowervalues[1:] - uppervalues[:-1]) * rising).sum() #sum where signal is rising

        falling = np.greater(lowervalues[:-1], uppervalues[1:])
        dummy += ((lowervalues[:-1] - uppervalues[1:]) * falling).sum() #sum where signal is falling
        
        numbox.append(dummy/blocksize[scale])

        if((numcols % 2) != 0): #double boxsize, half max and min
            uppervalues = (uppervalues[:-1].reshape(-1, 2)).max(axis=1)
            lowervalues = (lowervalues[:-1].reshape(-1, 2)).min(axis=1)
        else:
            uppervalues = (uppervalues.reshape(-1, 2)).max(axis=1)
            lowervalues = (lowervalues.reshape(-1, 2)).min(axis=1)

    N = np.log(numbox)
    R = np.log(1/np.array(blocksize))
    m = np.linalg.lstsq(R[:,None],N, rcond=None) #

    avg = np.average(entropy)
    std = np.std(entropy)
    mxe = np.max(entropy)
    mne = np.min(entropy)
    entdif = entropy[:-1] - entropy[1:]
    med = max(entdif.min(), entdif.max(), key=abs)
    frd = m[0][0]

    return np.array([avg, std, mxe, mne, med, frd])

def main():
    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [os.path.join(INPATH, x['path']) for x in d.values()]
    l = [x for x in l if os.path.isfile(x)]

    classes = set()
    for key, val in tqdm(d.items(), desc="build class set"):
        classes.add(val['track']['genre_top'])
    idx2lbl = dict(enumerate(classes))
    lbl2idx = {v:k for k,v in idx2lbl.items()}
    print(len(classes))

    target = [lbl2idx.get(x['track']['genre_top']) for x in d.values() if os.path.isfile(os.path.join(INPATH, x['path']))]
    print(len(target))
    del d

    pool = multiprocessing.Pool()
    imap = pool.imap(calc_entropy, l)
    l = [x for x in tqdm(imap, total=len(l))]
    print(len(l))

    outE = os.path.join(OUTPATH, 'entropy.npy')
    outL = os.path.join(OUTPATH, 'labels.npy')

    np.save(outE, np.array(l))
    np.save(outL, np.array(target))

if __name__ == '__main__':
    main()