#pragma once
#include "syk.hpp"
#include "syk_types.hpp"
#include "gpu_diagonalize.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <exception>
#include <algorithm>
#include <complex>

namespace syk {
using namespace std::complex_literals;

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
}