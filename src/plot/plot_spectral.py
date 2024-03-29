#!/usr/bin/env python3
import sys
import itertools
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd 
import flatbuffers
import time as ti
from numba import njit, prange
from SYKSchema.Output import Output
from SYKSchema.Checkpoint import Checkpoint

def get_ssf_from_file(filename, time, count = None):
    '''Attempt to read as output file and then as a checkpoint.
    Our buffers don't have identifiers for some reason so OutputBufferHasIdentifier always gives false.
    Fortunately, it blows up if we attempt to read it as Output and it's not.'''

    buffer = bytearray(open(filename, 'rb').read())

    try:
        output = Output.GetRootAsOutput(buffer, 0)
        assert(not output.DataIsNone())
        spectral = spectral_form_factor(output, time, count = count)
    except:
        output = Checkpoint.GetRootAsCheckpoint(buffer, 0).Output()
        assert(not output.DataIsNone())
        spectral = spectral_form_factor(output, time, count = count)

    return spectral
            

def spectral_form_factor(output, time, count = None):

    print('Total Compute: {}s'.format(output.TotalCompute()))

    @njit(nogil=True, parallel=True)
    def spectral_form_factor(time, data):
        g = np.zeros_like(time)
        for i in prange(len(time)):
            g[i] = np.mean(np.exp(data * time[i]))
        g = np.conj(g) * g
        return g

    spectral = np.zeros_like(time)

    num_points = output.DataLength()
    num_points = num_points if count is None else min(num_points, count)

    for i in range(num_points):
        point = output.Data(i)
        # Previously the RMT normalization was missing 1/sqrt(L)
        # spectral += spectral_form_factor(time, point.EigenvalsAsNumpy()/np.sqrt(point.EigenvalsAsNumpy().shape[0])) 
        spectral += spectral_form_factor(time, point.EigenvalsAsNumpy()) 

    spectral = spectral / num_points

    return spectral, num_points

def load_from_checkpoint(filename):
    checkpoint = Checkpoint.GetRootAsCheckpoint(bytearray(open(filename, 'rb').read()), 0)
    assert(not checkpoint.OutputIsNone())
    return checkpoint.output

def pretty_label(filename):
    '''Attempt to get a nice plot label from our naming convention'''
    try:
        parts = filename.split('/')[-1].split('.')
        assert(parts[-1] == 'bin')
        return parts[-2]
    except:
        return filename

if __name__ == '__main__':
    time = np.geomspace(1e-1, 1e6, 4000, dtype=np.dtype('c16'))*1.0j
    output_files = sys.argv[1:]


    colors = itertools.cycle(['b', 'r', 'c', 'm', 'k'])
    for filename in output_files:
        label = pretty_label(filename)
        print(f'Plotting {label}')

        timestart = ti.time()
        for count in [1,None]:
            spectral, points = get_ssf_from_file(filename, time, count=count)
            print(f'Plotted {label} in {ti.time() - timestart}')
            plt.yscale('log')
            plt.xscale('log')
            plt.plot(np.imag(time), np.real(spectral), label=f'{label}$~~N={points}$', color = next(colors))
    plt.grid()
    plt.legend(loc='lower right')
    plt.savefig('spectral.pdf')



