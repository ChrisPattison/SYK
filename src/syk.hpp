#pragma once
#include <cassert>
#include <algorithm>
#include <utility>
#include <complex>
#include <random>
#include <exception>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include "syk_types.hpp"
#include "util.hpp"

namespace syk {
using namespace std::complex_literals;

// Fermion and Pauli group representations

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

/** Utility class for construction Majorana fermion representations
 */
struct FermionRep {
    using PauliString = std::array<Pauli, 40>;
    using PauliMatrix = Eigen::Matrix<std::complex<short>, 2, 2>;
    int num_fermions_;
    
    PauliMatrix m_[4];


    FermionRep(int num_fermions) : num_fermions_(num_fermions) {
        assert(num_fermions > 0);
        assert(num_fermions % 2 == 0);

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

    Pauli PauliI() { return Pauli {AbstractPauli::I, Sign::P}; }
    Pauli PauliX() { return Pauli {AbstractPauli::X, Sign::P}; }
    Pauli PauliY() { return Pauli {AbstractPauli::Y, Sign::P}; }
    Pauli PauliZ() { return Pauli {AbstractPauli::Z, Sign::P}; }

    Pauli get_pauli_factor(int a, int qubit_index) {
        assert(0 <= a && a < num_fermions_);
        assert(0 <= qubit_index && qubit_index < num_fermions_/2);
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
        for(int qubit_index = 0; qubit_index < num_fermions_ / 2; ++qubit_index) {
            pstring[qubit_index] =
                  get_pauli_factor(a, qubit_index)
                * get_pauli_factor(b, qubit_index)
                * get_pauli_factor(c, qubit_index)
                * get_pauli_factor(d, qubit_index);
        }
        return pstring;
    }

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
        int max_level = num_fermions_/2 - 1;

        #pragma GCC unroll 8
        for(int level = 0; level <= max_level; ++level) {
            int mask = 1 << level;
            int sub_i = (i & mask) >> level;
            int sub_j = (j & mask) >> level;
            acc *= get_pauli_element(sub_i, sub_j, string[level]);
            if(acc.real() == 0 && acc.imag() == 0) { break; }
        }
        return zero_one_short_to_double(acc);
    }
};

/** Generate an SYK Hamiltonian
 */
template<typename rng_type>
__attribute__((optimize("fast-math")))
MatrixType syk_hamiltonian(rng_type* rng, int N, double J) {
    auto hilbert_space_size = (1 << (N+1)/2);
    MatrixType hamiltonian = MatrixType::Zero(hilbert_space_size, hilbert_space_size);
    auto repp = FermionRep(N);

    auto distr = std::normal_distribution<double>(0, J/std::pow(static_cast<double>(N), 1.5));

    std::vector<std::pair<FermionRep::PauliString, double>> interactions;
    interactions.reserve(N*N*N);
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < i; ++j) {
            for(int k = 0; k < j; ++k) {
                for(int l = 0; l < k; ++l) {
                    interactions.push_back(std::make_pair(repp.four_fermion(i, j, k, l), distr(*rng)));
                }
            }
        }
    }

    #pragma omp parallel for
    for(int j = 0; j < hamiltonian.rows(); ++j) {
        for(int i = 0; i < hamiltonian.cols(); ++i) {
            double matrix_element_re = 0.0;
            double matrix_element_im = 0.0;
            auto num_interactions = interactions.size();

            for(int k = 0; k < num_interactions; ++k) {
                auto single_element = repp.get_matrix_element(i, j, interactions[k].first);
                matrix_element_re += interactions[k].second * single_element.real();
                matrix_element_im += interactions[k].second * single_element.imag();
            }
            hamiltonian(i,j) = std::complex<double>(matrix_element_re, matrix_element_im);
        }
    }

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
