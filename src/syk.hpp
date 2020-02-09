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
#include <Eigen/Eigenvalues>
#include "syk_types.hpp"
#include "util.hpp"

#include "diagonalize.hpp"

namespace syk {
using namespace std::complex_literals;

enum class AbstractPauli {I = 0, X = 1, Y = 2, Z = 3};
enum class Sign {P = 0, M = 1};
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

int repp(Sign a) {
    return a == Sign::P ? 1 : -1;
}

MatrixType repp(AbstractPauli a) {
    MatrixType r(2,2);
    switch(a) {
        case AbstractPauli::I:
            r << 1, 0,
                 0, 1; break;
        case AbstractPauli::X:
            r << 0, 1,  
                 1, 0; break;
        case AbstractPauli::Y:
            r << 0, -1.0i,
                 1.0i, 0; break;
        case AbstractPauli::Z:
            r << 1, 0,
                 0, -1; break;
    }
    return r;
}

struct FermionRep {
    int num_fermions_;

    FermionRep(int num_fermions) : num_fermions_(num_fermions) {
        assert(num_fermions > 0);
        assert(num_fermions % 2 == 0);
    }
    
    auto get_repp_factor(Pauli a) {
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

    auto four_fermion(int a, int b, int c, int d, std::complex<double> coeff) {
        MatrixType output = coeff * MatrixType::Identity(1,1);

        for(int qubit_index = 0; qubit_index < num_fermions_/2; ++qubit_index) {
            auto factor = get_pauli_factor(a, qubit_index)
                        * get_pauli_factor(b, qubit_index)
                        * get_pauli_factor(c, qubit_index)
                        * get_pauli_factor(d, qubit_index);
            MatrixType repp_factor = get_repp_factor(factor);
            output = Eigen::kroneckerProduct(output, repp_factor).eval();
        }
        return output;
    }
};

template<typename rng_type>
MatrixType syk_hamiltonian(rng_type* rng, int N, double J) {
    assert(N%2 == 0);
    auto hilbert_space_size = (1 << N/2);
    MatrixType hamiltonian = MatrixType::Zero(hilbert_space_size, hilbert_space_size);
    auto repp = FermionRep(N);

    auto distr = std::normal_distribution<double>(0, J/std::pow(static_cast<double>(N), 1.5));

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < i; ++j) {
            for(int k = 0; k < j; ++k) {
                for(int l = 0; l < k; ++l) {
                    auto interaction = repp.four_fermion(i, j, k, l, distr(*rng));
                    hamiltonian = (hamiltonian + interaction).eval();
                }
            }
        }
    }
    return hamiltonian;
}

std::vector<double> cpu_hamiltonian_eigenvals(const MatrixType& hamiltonian) {
    auto eigensolver = Eigen::SelfAdjointEigenSolver<MatrixType>();
    eigensolver.compute(hamiltonian, false);

    if(eigensolver.info() != 0) {
        throw std::runtime_error("Eigensolver not converged");
    }
    const auto& eigenvals_vector = eigensolver.eigenvalues();
    std::vector<double> eigenvals(eigenvals_vector.size());
    std::copy_n(eigenvals_vector.data(), eigenvals_vector.size(), eigenvals.begin());
    std::sort(eigenvals.begin(), eigenvals.end());

    return eigenvals;
}

std::vector<double> gpu_hamiltonian_eigenvals(const MatrixType& hamiltonian) {
    syk::GpuEigenValSolver solver;
    std::vector<double> eigenvals;
    #pragma omp critical
    eigenvals = solver.eigenvals(hamiltonian);
    return eigenvals;
}


auto spectral_form_factor(const std::vector<double>& eigenvals) {
    return [=](std::complex<double> beta) -> double {
        auto z_part = util::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0i,
            std::plus<>(), [=](auto a) { return std::exp(a * beta); });
        return std::real(z_part * std::conj(z_part));
    };
}
}
