import os
import pydub
import scipy
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

print(fft.shape)

# print(rate)
# print(data.shape)
# print(data)

plt.plot(fft)
plt.show()