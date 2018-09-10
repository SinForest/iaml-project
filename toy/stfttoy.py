import os
import pydub
import scipy
from scipy import signal
from scipy.io import wavfile
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from csv import reader

LEFT = 0

seg = pydub.AudioSegment.from_mp3("/home/cobalt/datasets/fma/fma_small/000/000002.mp3") 
f = tempfile.NamedTemporaryFile()
seg.export(f.name, format='wav')
rate, data = wavfile.read(f.name)
f.close()

part = data[:rate*30, LEFT]
fft = np.fft.fft(part)
#stft = signal.stft(part)

#print(stft.shape)

# print(rate)
# print(data.shape)
# print(data)

f, t, Zxx = signal.stft(part, rate, nperseg=2**13)

hist, bin_edges = np.histogram(Zxx, bins=1000)

print(Zxx.shape)
exit()

plt.bar(range(len(hist)), hist)
plt.xticks(range(len(hist)), bin_edges)

#plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=20)


#plt.plot(fft)
plt.show()