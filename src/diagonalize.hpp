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
#include "syk.hpp"
#include "syk_types.hpp"
#include "cuda_diagonalize.hpp"

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

template<typename mat> 
std::complex<double> closest_eigenval(mat m, std::complex<double> v) {
    std::complex<double> a, b, c, d;
    a = m(0,0);
    b = m(0,1);
    c = m(1,0);
    d = m(1,1);
    
    auto desc = std::sqrt(a*a + 4.0 * b * c - 2.0 * a * d + d * d);
    auto l_1 = (a + d + desc)/2.0;
    auto l_2 = (a + d - desc)/2.0;
    
    auto dist_1 = std::abs(l_1 - v);
    auto dist_2 = std::abs(l_2 - v);
    return dist_1 < dist_2 ? l_1 : l_2;
}

std::vector<double> QR_hamiltonian_eigenvals(const MatrixType& ham, MatrixType* Q_H,  double max_resid = 1e-7) {
    int max_iter = 1000 * ham.rows();
    assert(ham.rows() == ham.cols());
    assert(Q_H->rows() == ham.rows());
    assert(Q_H->cols() == ham.cols());

    // Eigen::HessenbergDecomposition<MatrixType> hd;
    // hd.compute(ham);
    // *Q_H = hd.matrixQ().adjoint();

    MatrixType temp(ham.rows(), ham.cols());
    MatrixType Q_curr(ham.rows(), ham.cols());
    
    MatrixType A = Q_H->adjoint() * ham * *Q_H;
    Eigen::HouseholderQR<MatrixType> qr(ham.rows(), ham.cols());
    
    // TODO: Use tridiagonal form
    int k;
    int deflate = ham.diagonal().size();
    for(k = 0; k < max_iter; ++k) {
        // // Rayleigh quotient
        // std::complex<double> shift = A(deflate-1, deflate-1);
        // Wilkinson
        auto shift = closest_eigenval(A.block(deflate-2, deflate-2, 2, 2), A(deflate-1, deflate-1));
        // QR
        qr.compute(A.topLeftCorner(deflate, deflate) - MatrixType::Identity(deflate, deflate) * shift);
        Q_curr = qr.householderQ();
        // Update Q_H with deflated matmul
        temp.topLeftCorner(deflate, deflate).noalias() = Q_H->topLeftCorner(deflate, deflate) * Q_curr;
        temp.bottomLeftCorner(Q_H->rows() - deflate, deflate).noalias() = Q_H->bottomLeftCorner(Q_H->rows() - deflate, deflate) * Q_curr;
        Q_H->topLeftCorner(Q_H->rows(), deflate) = temp.topLeftCorner(Q_H->rows(), deflate);
        // Update A = Q^* H Q
        A.noalias() = Q_H->adjoint() * ham * *Q_H; // TODO: Deflate this
        // Check convergence condition
        if(std::abs(A(deflate-1, deflate-2)) < max_resid && --deflate == 1) {
            break;
        }
    }

    double residual = 0;
    for(int j = 0; j < A.rows(); ++j) {
        for(int i = j+1; i < A.cols(); ++i) {
            residual += std::real(A(i,j) * std::conj(A(i,j)));
        }
    }
    residual = std::sqrt(residual) / A.diagonal().size();
    
    std::vector<double> eigenvals(A.diagonal().size());
    Eigen::Map<Eigen::VectorXd>(eigenvals.data(), eigenvals.size()) = A.diagonal().real();
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