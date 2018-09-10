import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from chrono import Timer

dur = 31 # 168 # hardcoded: should be d[2]["track"]["duration"]
offs = float(np.random.choice(dur - 30))
with Timer() as timed:
    y, sr = librosa.load("/home/cobalt/datasets/fma/fma_small/000/000002.mp3", mono=True, duration=30., offset=offs)
print(f"[TIME] loading: {timed.elapsed}")

print(f"offset: {offs}; length of y: {len(y)}")

hop_length = 2*11

print("### computing mel spectrogram ###")
with Timer() as timed:
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=64, n_fft=2**12, hop_length=hop_length)
print(f"[TIME] mel: {timed.elapsed}")
# Convert to log scale (dB). We'll use the peak power (max) as reference.
log_S = librosa.power_to_db(S, ref=np.max)

print("### computing onset envelope ###")
with Timer() as timed:
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
print(f"[TIME] onset: {timed.elapsed}")
print("### computing tempogram ###")
with Timer() as timed:
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
print(f"[TIME] tempo: {timed.elapsed}")
print("### BREAKPOINT ###")
breakpoint()

# TODO: test different features, benchmark in generation time
# TODO: make everything shorter! 