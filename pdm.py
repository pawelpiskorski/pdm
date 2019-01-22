import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt



fs = 48000
Ts = 2*np.pi/fs
nyq = fs/2
upsample = 64

duration = 1  # seconds
time = np.arange(0, duration*fs, 1)
s = 0.7*np.sin(4027*Ts*time) +0.3*np.sin(8013*Ts*time)  # signal to encode
length = s.shape[0]
np.random.seed(19259)

# encode

pdm = signal.resample(s, length*upsample)
Rnyq = upsample*nyq


# noise shaped to occupy frequency band beyond the signal
# (since after upsampling maximum frequency is upsample*nyq)
b, a = signal.butter(3, 1.2*nyq/Rnyq, btype='high', analog=False)
y = signal.filtfilt(b, a, np.random.normal(0., 1., pdm.shape[0]))

# quantize signal to just -1 or 1 depending on sign.
# note: happily ignore that np.sign(0.) == 0.
pdm = np.sign(pdm+y)


# decode: low pass to original band, and resample
b, a = signal.butter(5, nyq/Rnyq, btype='low', analog=False)
d = signal.filtfilt(b, a, pdm)
d = signal.resample(d, length)


def plot(orig, pdm, decoded, time, Ts=Ts, upsample=upsample):
	cursor=2000
	plen = 20
	plt.subplot(3,1,1)
	plt.plot(Ts*time[cursor:cursor+plen], orig[cursor:cursor+plen])
	plt.title("Original signal")
	plt.subplot(3,1,2)
	plt.plot(Ts/upsample*np.arange(plen*upsample), pdm[cursor*upsample:(cursor+plen)*upsample], linewidth=0.4)
	plt.title(f"{upsample}x upsampled, PDM encoded")
	plt.subplot(3,1,3)
	plt.plot(Ts*time[cursor:cursor+plen], decoded[cursor:cursor+plen])
	plt.title("Decoded signal")
	plt.tight_layout()
	plt.show()

plot(s, pdm, d, time)