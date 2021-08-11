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
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>
#include <functional>

namespace util {

template<typename T>
std::vector<T> logspace(T a, T b, int N) {
    int n = 0;
    std::vector<T> vals(N);
    auto m = (std::log(b) - std::log(a)) / static_cast<T>(N);
    std::generate(vals.begin(), vals.end(), [&]() {
        return std::exp(m * static_cast<T>(n++) + std::log(a)); });
    return vals;
}

template<typename T>
std::vector<T> linspace(T a, T b, int N) {
    int n = 0;
    std::vector<T> vals(N);
    auto m = (b - a) / static_cast<T>(N);
    std::generate(vals.begin(), vals.end(), [&]() {
        return m * static_cast<T>(n++) + a; });
    return vals;
}

// http://blog.geomblog.org/2005/10/sampling-from-simplex.html
template<typename rng_type>
std::vector<double> sample_unit_simplex(rng_type* rand_gen, int dim) {
    assert(dim >= 2);
    std::uniform_real_distribution distr;
    std::vector<double> vals(dim+2);
    std::generate(vals.begin()+2, vals.end(), [&]() { return distr(*rand_gen); });
    vals[0] = 0;
    vals[1] = 1;
    std::sort(vals.begin(), vals.end());
    std::adjacent_difference(vals.begin(), vals.end(), vals.begin());
    vals.erase(vals.begin());
    assert(std::abs(std::accumulate(vals.begin(), vals.end(), 0.0) - 1.0) < 1e-7);
    return vals;
}

template<typename rng_type>
std::vector<double> sample_biased_simplex(rng_type* rand_gen, int dim, double size) {
    auto simplex = sample_unit_simplex(rand_gen, dim);
    std::transform(simplex.begin()+1, simplex.end(), simplex.begin(), [&](const auto& v) { return v * size; });
    assert(size < 1.0);
    auto norm = std::accumulate(simplex.begin(), simplex.end(), 0.0);
    std::transform(simplex.begin(), simplex.end(), simplex.begin(), [&](const auto& v) { return v / norm; });
    return simplex;
}

template<typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
T transform_reduce(InputIt first, InputIt last, T init, BinaryOp binop, UnaryOp unary_op) {
    T result = init;
    for(; first != last; ++first) {
        result = binop(result, unary_op(*first));
    }
    return result;
}


template <typename InputIt1, typename InputIt2, typename T, typename BinaryOp1, typename BinaryOp2>
T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2) {
    T result = init;
    while(first1 != last1) {
        result = binary_op1(result, binary_op2(*first1, *first2));

        ++first1;
        ++first2;
    }
    return result;
}

template<typename InputIt1, typename InputIt2, typename T>
T transform_reduce(InputIt1 first1, InputIt1 last1, InputIt2 first2, T init) {
    return transform_reduce(first1, last1, first2, init, std::plus<>(), std::multiplies<>());
}

template<typename rng_type>
void warmup_rng(rng_type* rng, int warmup_cycles = 100000) {
    for(int k = 0; k < warmup_cycles; ++k) {
        (*rng)();
    }
}

template<typename stream_type>
std::size_t get_stream_size(stream_type* stream) {
    stream->seekg(0, std::ios::end);
    std::size_t num_bytes = stream->tellg();
    stream->seekg(0, std::ios::beg);
    return num_bytes;
}
}