import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.fft import fft, ifft, fftfreq

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('A', type=np.double)
argparser.add_argument('fmax', type=np.double)
argparser.add_argument('T', type=np.double)
argparser.add_argument('NoPoints', type=int)
argparser.add_argument('filename', type=str)
argparser.add_argument('--seed', nargs='?', type=int)
args = argparser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

Fs = 2 * args.NoPoints/args.T
dt1 = 1 / Fs
t1 = np.arange(0, args.T, dt1)
y1 = 2 * (np.random.uniform(size=np.size(t1)) - .5) * np.pi

dt2 = 1 / (2 * args.fmax)
t2 = np.arange(0, args.T - dt2, dt2)
y2 = np.interp(t2, t1, y1)
y2[0] = 0
y2[-1] = 0

NFFT1 = np.size(y1)
Y1 = fft(y1, NFFT1) / NFFT1
f1 = Fs / 2 * np.linspace(0, 1, int(NFFT1/2))

NFFT2 = np.size(y2)
Y2 = fft(y2, NFFT2) / NFFT2
Fs2 = 1 / dt2
f2 = Fs2 / 2 * np.linspace(0, 1, int(NFFT2/2))

f4 = f1
temp = fft(y2)
temp = temp[1:int(np.size(temp)/2)]
Y4 = np.concatenate((temp, np.zeros(np.size(t1) - np.size(t2))), axis=0)

y4 = np.real(ifft(Y4, np.size(f4)))
y4 /= max(abs(y4))
y4 *= args.A

if (args.filename.split('.')[-1] == 'csv'):
    np.savetxt(args.filename, y4, delimiter=',')
else:
    np.savetxt(args.filename + '.csv', y4, delimiter=',')

