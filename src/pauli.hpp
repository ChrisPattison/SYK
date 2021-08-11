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
#include <cstdint>
#include <exception>

#include <Eigen/Dense>
#include "syk_types.hpp"
#include "util.hpp"

/** Utilities for working with a Hamiltonian as a sum of Pauli operators
 */
namespace syk {
using namespace std::complex_literals;

/** Pauli group
 */

enum class AbstractPauli : short {I = 0, X = 1, Y = 2, Z = 3};
enum class Sign : short {P = 1, M = -1};
struct Pauli { AbstractPauli m; Sign s; };

Sign operator*(Sign a, Sign b) {
    return (a == b) ? Sign::P : Sign::M;
}

Sign operator!(Sign a) {
    return a * Sign::M;
}

Pauli operator*(Pauli a, Pauli b) {
    auto new_sign = a.s * b.s;
    if( !(static_cast<int>(a.m) <= static_cast<int>(b.m)) ) {
        std::swap(a, b);
        new_sign = (
                a.m == AbstractPauli::I 
            ||  b.m == AbstractPauli::I
            ||  a.m == b.m) ? new_sign : !new_sign;
    }
    
    if (a.m == AbstractPauli::I) { return Pauli {b.m, new_sign}; }
    if (a.m == b.m) { return Pauli {AbstractPauli::I, new_sign}; }
    return Pauli {
        static_cast<AbstractPauli>(3 - (static_cast<int>(a.m) + static_cast<int>(b.m))%3),
        new_sign};
}

/** Utilities for construction Hamiltonian as a sum of products of Paulis
 * This has a hard coded maximum number of qubits (20) for performance
 * If/When increased it should be a multiple of 4 to allow vectorization
 * 
 * Constructing the full Hamiltonian is done coefficient-wise by going through each term in the sum
 * For each term we stop when we hit a 0
 * This is branch heavy but was about 2x faster than without the branch
 */
struct PauliRep {
    static constexpr int kmax_qubits = 20;
    using PauliString = std::array<Pauli, kmax_qubits>;
    using PauliMatrix = Eigen::Matrix<std::complex<short>, 2, 2>;

    struct Term {
        PauliString pauli_op;
        double weight;
    };

protected:
    int num_qubits_;

public:
    PauliMatrix m_[4];

    PauliRep(int num_qubits) : num_qubits_(num_qubits) {
        if(num_qubits > kmax_qubits) { throw std::runtime_error("Number of qubits exceeds maximum support. Increase the compile time value"); }

        m_[0] << 1, 0,
                0, 1;

        m_[1] << 0, 1,  
                1, 0;

        m_[2] << 0, -1i,
                1i, 0;

        m_[3] << 1, 0,
                0, -1;
    }

    /** Get sign
     */
    short repp(Sign a) const __attribute__((always_inline)) {
        return static_cast<short>(a);
    }

    /** Get matrix for a one qubit Pauli
     */
    const PauliMatrix& repp(AbstractPauli a) const __attribute__((always_inline)) {
        return m_[static_cast<short>(a)];
    }

    auto get_repp_factor(Pauli a) const __attribute__((always_inline)) {
        return repp(a.m) * repp(a.s);
    }

    Pauli PauliI() const { return Pauli {AbstractPauli::I, Sign::P}; }
    Pauli PauliX() const { return Pauli {AbstractPauli::X, Sign::P}; }
    Pauli PauliY() const { return Pauli {AbstractPauli::Y, Sign::P}; }
    Pauli PauliZ() const { return Pauli {AbstractPauli::Z, Sign::P}; }

    /** Single Pauli matrix element
     */
    std::complex<short> get_pauli_element(int i, int j, Pauli p) const __attribute__((always_inline)) {
        return repp(p.m)(i,j) * repp(p.s);
    }

    /** Convert a short taking values 0, +/- 1 to double
     */
    std::complex<double> zero_one_short_to_double(std::complex<short> a) const {
        return std::complex<double>(
            a.real() == 0 ? 0.0 : (a.real() > 0 ? 1.0 : -1.0),
            a.imag() == 0 ? 0.0 : (a.imag() > 0 ? 1.0 : -1.0)
        );
    }

    /** Get matrix element of a string of Paulis
     * shorts everywhere to hopefully let the compiler do its thing
     */
    std::complex<double> get_matrix_element(int i, int j, const PauliString& string) {
        std::complex<short> acc(1,0);
        int max_level = num_qubits_ - 1;

        #pragma GCC unroll 4
        for(int level = 0; level <= max_level; ++level) {
            int mask = 1 << level;
            int sub_i = (i & mask) >> level;
            int sub_j = (j & mask) >> level;
            acc *= get_pauli_element(sub_i, sub_j, string[level]);
            if(acc.real() == 0 && acc.imag() == 0) { break; }
        }
        return zero_one_short_to_double(acc);
    }

    /** Compute the Hamiltonian for some sum of terms
     * Give only the cols/rows specified by subspace
     * Subspace is usually 0 to (2^N - 1) but could be something else where symmetries exist
     */
    MatrixType get_hamiltonian(std::vector<Term> terms, std::vector<std::uint64_t> subspace) {
        MatrixType hamiltonian = MatrixType::Zero(subspace.size(), subspace.size());

        #pragma omp parallel for
        for(int j = 0; j < subspace.size(); ++j) {
            for(int i = 0; i < subspace.size(); ++i) {
                double matrix_element_re = 0.0;
                double matrix_element_im = 0.0;
                auto num_terms = terms.size();

                for(int k = 0; k < num_terms; ++k) {
                    auto single_element = get_matrix_element(subspace[i], subspace[j], terms[k].pauli_op);
                    matrix_element_re += terms[k].weight * single_element.real();
                    matrix_element_im += terms[k].weight * single_element.imag();
                }
                hamiltonian(i,j) = std::complex<double>(matrix_element_re, matrix_element_im);
            }
        }

        return hamiltonian;
    }
};
}