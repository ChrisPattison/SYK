import sys
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import flatbuffers
from numba import njit, prange
from SYKSchema.Output import Output

output = Output.GetRootAsOutput(bytearray(open(sys.argv[1], 'rb').read()), 0)
assert(not output.DataIsNone())

@njit(nogil=True, parallel=True)
def spectral_form_factor(time, data):
    g = np.zeros_like(time)
    for i in prange(len(time)):
        g[i] = np.sum(np.exp(data * time[i]))
    g = np.conj(g) * g
    return g

time = np.geomspace(1e-1, 1e6, 4000, dtype=np.dtype('c16'))*1.0j

spectral = np.zeros_like(time)

for point in (output.Data(i) for i in range(output.DataLength())):
    spectral += spectral_form_factor(time, point.EigenvalsAsNumpy())

spectral = spectral / (output.DataLength()**2)
plt.yscale('log')
plt.xscale('log')
plt.plot(np.imag(time), np.real(spectral))
plt.savefig('plot.pdf')

