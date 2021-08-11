/* Copyright (c) 2020 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
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