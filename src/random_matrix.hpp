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
 
#pragma once
#include <cassert>
#include <algorithm>
#include <utility>
#include <complex>
#include <random>
#include <cmath>

#include <Eigen/Dense>
#include "syk_types.hpp"

// https://web.eecs.umich.edu/~rajnrao/Acta05rmt.pdf

namespace syk {
using namespace std::complex_literals;

template<typename rng_type>
MatrixType RandomRealIidGauss(rng_type* rand_gen, int n, int m) {
    MatrixType mat = MatrixType::Zero(n, m);
    std::normal_distribution normal;

    for(int j = 0; j < m; ++j) {
        for(int i = 0; i < n; ++i) {
            mat(i, j) = static_cast<MatrixType::Scalar>(normal(*rand_gen));
        }
    }
    return mat;
}

template<typename rng_type>
MatrixType RandomGOE(rng_type* rand_gen, int n) {
    MatrixType A = RandomRealIidGauss(rand_gen, n, n);
    return (A + A.transpose()) / (2 * std::sqrt(n));
}

template<typename rng_type>
MatrixType RandomGUE(rng_type* rand_gen, int n) {
    MatrixType A = RandomRealIidGauss(rand_gen, n, n) + 1.0i * RandomRealIidGauss(rand_gen, n, n);
    return (A + A.adjoint()) / (2 * std::sqrt(n));
}
}