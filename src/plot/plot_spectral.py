#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import flatbuffers
import time as ti
from spectral_form_factor import spectral_form_factor
from SYKSchema.Output import Output

output = Output.GetRootAsOutput(bytearray(open(sys.argv[1], 'rb').read()), 0)
assert(not output.DataIsNone())

print('Total Compute: {}s'.format(output.TotalCompute()))

time = np.geomspace(1e-1, 1e6, 1000, dtype=np.dtype('d'))

time_start = ti.time()
# 10K x 1024 x 1000
stacked_eigenvals = []
for point in (output.Data(i) for i in range(output.DataLength())):
    stacked_eigenvals.append(point.EigenvalsAsNumpy())
# Sample index is axis 0
# Eigenval index is axis 1
stacked_eigenvals = np.concatenate(stacked_eigenvals, axis=0)
spectral = spectral_form_factor(time, stacked_eigenvals)

print('Spectral form factor done in {}s'.format(ti.time() - time_start))

spectral = spectral
plt.yscale('log')
plt.xscale('log')
plt.plot(time, spectral)
plt.savefig('plot.pdf')

