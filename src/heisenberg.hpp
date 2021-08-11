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
#include "pauli.hpp"
#include <vector>
#include <random>

namespace syk  {
PauliRep::Term two_body(double weight, int i, int j, Pauli op_i, Pauli op_j) {
    PauliRep::Term term;
    term.weight = weight;
    term.pauli_op.fill(Pauli {AbstractPauli::I, Sign::P});
    term.pauli_op[i] = op_i;
    term.pauli_op[j] = op_j;
    return term;
}

std::vector<PauliRep::Term> heisenberg_spin_chain(int N) {
    std::vector<PauliRep::Term> chain;
    chain.reserve(3*N);

    auto X = Pauli {AbstractPauli::X, Sign::P};
    auto Y = Pauli {AbstractPauli::Y, Sign::P};
    auto Z = Pauli {AbstractPauli::Z, Sign::P};
    
    for(int i = 0; i < N; ++i) {
        int site = i;
        int next_site = (i+1)%N;
        chain.push_back(two_body(0.25, site, next_site, X, X));
        chain.push_back(two_body(0.25, site, next_site, Y, Y));
        chain.push_back(two_body(0.25, site, next_site, Z, Z));
    }

    return chain;
}

/** Random longitudinal uniformly distributed in [-W, +W] 
 */
template<typename rng_type>
std::vector<PauliRep::Term> random_field(rng_type* rng, int N, double W) {
    std::vector<PauliRep::Term> terms;

    terms.reserve(N);
    auto Z = Pauli {AbstractPauli::Z, Sign::P};

    auto distr = std::uniform_real_distribution(-W, W);

    for(int i = 0; i < N; ++i) {
        PauliRep::Term single_term;
        single_term.weight = distr(*rng);
        single_term.pauli_op.fill(Pauli {AbstractPauli::I, Sign::P});
        single_term.pauli_op[i] = Z;

        terms.push_back(single_term);
    }
    
    return terms;
}

/** Heisenberg spin chain with random longitudinal field
 * Limited to total spin 0 (1/2) subspace for even (odd) number of qubits
 */
template<typename rng_type>
MatrixType random_field_heisenberg(rng_type* rng, int N, double W) {
    auto spin_chain = heisenberg_spin_chain(N);
    auto fields = random_field(rng, N, W);
    spin_chain.insert(spin_chain.end(), fields.begin(), fields.end());

    std::uint64_t full_subspace = 2UL<<N;
    int hamming_weight = N/2;
    std::vector<std::uint64_t> subspace;
    subspace.reserve(full_subspace);
    for(std::uint64_t k = 0; k < full_subspace; ++k) {
        if(__builtin_popcount(k) == hamming_weight) {
            subspace.push_back(k);
        }
    }

    auto hamiltonian = PauliRep(N).get_hamiltonian(spin_chain, subspace);
    return hamiltonian;
}
}