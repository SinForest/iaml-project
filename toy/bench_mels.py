import librosa

song, sr = librosa.load("./dataset/000/000002.mp3", mono=True, duration=30)
d = {}

for fft_exp in [8,9,10,11,12,13,14,15]:
    d[fft_exp] = {}
    for hop_exp in [8,9,10,11,12,13,14,15]:
        if hop_exp > fft_exp:
            continue
        overlap = (1-1/2**(fft_exp - hop_exp))
        frame_dur = (2**fft_exp / sr)
        if overlap > 0.9:
            continue
        
        X = librosa.feature.melspectrogram(song, sr=sr, n_mels=64, n_fft=2**fft_exp, hop_length=2**hop_exp)

        if not 200 <= X.shape[1] <= 2000:
            continue

        print(f"fft: {fft_exp:>2}; hop: {hop_exp:>2} => {X.shape[1]:>5} "\
              f"(dur: {frame_dur:2.2f}s; over: {overlap*100:3.2f}%)")
