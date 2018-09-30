import h5py
import librosa
import numpy as np
import os
import pickle
import multiprocessing
from tqdm import tqdm

INPATH  = "./dataset.ln"
OUTPATH = "./melsset.ln"

def save_mel(paths):
    
    try:
        inpath, outpath = paths

        song, sr = librosa.load(inpath, mono=True)

        n_fft = 2**11         # shortest human-disting. sound (music)
        hop_length = 2**10    # => 50% overlap of frames
        n_mels = 128

        X = librosa.feature.melspectrogram(song, sr=sr, n_fft=n_fft, hop_length=hop_length)
        if not os.path.isdir(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        
        np.save(outpath, X.astype(np.float32))
    except:
        return 1

    return 0

def main():
    d = pickle.load(open("./all_metadata.p", 'rb'))
    l = [(os.path.join(INPATH, x['path']), os.path.join(OUTPATH, x['path'][:-4] + '.npy')) for x in d.values()]
    del d

    pool = multiprocessing.Pool()

    imap = pool.imap(save_mel, l)
    ssss = sum([x for x in tqdm(imap)])

    print(f"finished, {ssss} errors occured")
    


    #print(save_mel(("/share/fma_full/000/000002.mp3", "/scratch/fma_mels/000/000002.npy")))

if __name__ == '__main__':
    main()
