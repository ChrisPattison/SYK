import numpy as np
from scipy.special import roots_hermite
import itertools
import sympy as sp
import sys

# Coefficients for Gauss-Hermite quadrature with truncation

order = 16000
trunc = 1e-10

x, w = roots_hermite(order)

stacked = np.stack([x, w], axis=0)
stacked = stacked[:, np.abs(w) > trunc]

x = stacked[0, :]
w = stacked[1, :]

print(len(x), file=sys.stderr)

codegen = []
codegen.append(r'#include <vector>')
codegen.append('')
codegen.append(r'namespace util {')

codegen.append(r'std::vector<double> gauss_hermite_points  = {{ {} }};'.format(', '.join(str(v) for v in x)))
codegen.append(r'std::vector<double> gauss_hermite_weights = {{ {} }};'.format(', '.join(str(v) for v in w)))

codegen.append(r'}')
print('\n'.join(codegen))