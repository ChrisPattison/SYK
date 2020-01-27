#pragma once
#include <cassert>
#include <algorithm>
#include <utility>
#include <complex>
#include <random>
#include <cmath>

#include <eigen3/Eigen/Dense>
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
    auto A = RandomRealIidGauss(rand_gen, n, n);
    return (A + A.transpose()) / 2;
}

template<typename rng_type>
MatrixType RandomGUE(rng_type* rand_gen, int n) {
    auto A = RandomRealIidGauss(rand_gen, n, n) + 1.0i * RandomRealIidGauss(rand_gen, n, n);
    return (A + A.adjoint()) / 2;
}
}