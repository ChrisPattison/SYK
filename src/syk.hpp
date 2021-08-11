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
#include <exception>
#include <cmath>

#include <Eigen/Dense>
#include "syk_types.hpp"
#include "pauli.hpp"
#include "util.hpp"

/** Generation for SYK Hamiltonians
 */

namespace syk {
using namespace std::complex_literals;

/** Utility class for construction Majorana fermion representations
 */
struct FermionRep : PauliRep {
protected:
    int num_fermions_;

public:

    FermionRep(int num_fermions) : PauliRep(num_fermions/2), num_fermions_(num_fermions) {
        if(!(num_fermions > 0 && num_fermions % 2 == 0)) { throw std::runtime_error("Number of fermions must be positive and even"); }
    }

    Pauli get_pauli_factor(int a, int qubit_index) {
        assert(0 <= a && a < num_fermions_);
        assert(0 <= qubit_index && qubit_index < num_qubits_);
        if (a/2 == qubit_index) { return (a%2 == 1) ? PauliY() : PauliX(); }
        if (a/2 < qubit_index)  { return PauliI(); }
        if (a/2 > qubit_index)  { return PauliZ(); }
        
        assert(false);
        return PauliI();
    }

    /** Four body interaction terms
     */
    PauliString four_fermion(int a, int b, int c, int d) {
        PauliString pstring;
        for(int qubit_index = 0; qubit_index < num_qubits_; ++qubit_index) {
            pstring[qubit_index] =
                  get_pauli_factor(a, qubit_index)
                * get_pauli_factor(b, qubit_index)
                * get_pauli_factor(c, qubit_index)
                * get_pauli_factor(d, qubit_index);
        }
        return pstring;
    }

    /** Returns the parity operator (gamma_*) for the Clifford algebra repp
     */
    PauliString gamma_star() const {
        PauliString pstring;
        pstring.fill(PauliZ());
        return pstring;
    }
};

/** Return all numbers of even hamming weight below size
 */
std::vector<std::uint64_t> hamming_weight(std::uint64_t size, bool odd = false) {

    std::vector<std::uint64_t> idx;
    idx.reserve(size/2);
    for (std::uint64_t i = 0; i < size; ++i) {
        if (static_cast<bool>(__builtin_popcount(i) % 2) == odd) {
            idx.push_back(i);
        }
    }

    return idx;
}

/** Generate an SYK Hamiltonian
 * Returns only a single parity sector
 */
template<typename rng_type>
__attribute__((optimize("fast-math")))
MatrixType syk_hamiltonian(rng_type* rng, int N, double J) {
    auto hilbert_space_size = (1 << (N+1)/2);
    auto repp = FermionRep(N);

    auto distr = std::normal_distribution<double>(0, J/std::pow(static_cast<double>(N), 1.5));

    std::vector<FermionRep::Term> interactions;
    interactions.reserve(N*N*N);
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < i; ++j) {
            for(int k = 0; k < j; ++k) {
                for(int l = 0; l < k; ++l) {
                    interactions.push_back(FermionRep::Term { repp.four_fermion(i, j, k, l), distr(*rng)} );
                }
            }
        }
    }

    // The parity projection (1 + gamma_*) removes all rows/columns with odd hamming weight
    auto even_weight = hamming_weight(hilbert_space_size);
    auto hamiltonian = repp.get_hamiltonian(interactions, even_weight);

    return hamiltonian;
}

/** Returns a lambda for computing the spectral form factor at some time
 */
auto spectral_form_factor(const std::vector<double>& eigenvals) {
    return [=](std::complex<double> beta) -> double {
        auto z_part = util::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0i,
            std::plus<>(), [=](auto a) { return std::exp(a * beta); });
        return std::real(z_part * std::conj(z_part));
    };
}

/** Vector of doubles to Eigen::VectorXd
 */
auto to_eigen_vector(const std::vector<double>& vals) {
    Eigen::Matrix<MatrixType::Scalar, Eigen::Dynamic, 1> vec(vals.size());
    std::copy(vals.begin(), vals.end(), vec.data());
    return vec;
}
}
