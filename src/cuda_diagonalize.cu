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
 
#include "cuda_diagonalize.hpp"

#include <cassert>
#include <algorithm>
#include <complex>
#include <exception>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


namespace syk {

#if DIAG_SINGLE_PRECISION
    using float_t = float;
    using cuComplex_t = cuFloatComplex;
    #define PRECISION_HEEVD_bufferSize cusolverDnCheevd_bufferSize
    #define PRECISION_HEEVD cusolverDnCheevd
    #define PRECISION_MAKE_FLOAT2 make_float2
#else
    using float_t = double;
    using cuComplex_t = cuDoubleComplex;
    #define PRECISION_HEEVD_bufferSize cusolverDnZheevd_bufferSize
    #define PRECISION_HEEVD cusolverDnZheevd
    #define PRECISION_MAKE_FLOAT2 make_double2
#endif


GpuEigenValSolver::GpuEigenValSolver() {
    if(cusolverDnCreate(&handle_)) { throw std::runtime_error("Failed to init cusolver"); }
}

GpuEigenValSolver::~GpuEigenValSolver() {
    cusolverDnDestroy(handle_);
}


// std::vector<double> GpuEigenValSolver::eigenvals(const MatrixType& matrix) {
//     assert(matrix.cols() == matrix.rows());
//     int cols = matrix.cols();
//     thrust::device_vector<double> result(cols);

//     thrust::host_vector<cuDoubleComplex> host_A(matrix.size());
//     std::transform(matrix.data(), matrix.data() + matrix.size(), host_A.begin(), [](auto v) { return make_double2(v.real(), v.imag()); });
//     thrust::device_vector<cuDoubleComplex> A = host_A;
//     cudaDeviceSynchronize();

//     // Work size query
//     int lwork;
//     auto status = cusolverDnZheevd_bufferSize(handle_, 
//         CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, 
//         cols, thrust::raw_pointer_cast(A.data()), cols, thrust::raw_pointer_cast(result.data()), &lwork);
//     cudaDeviceSynchronize();
//     if(status) { throw std::runtime_error("Failed to get cusolver buffer size"); }

//     // Diagonalize
//     // TODO: Check storage order and if zgeev is expecting square matrix 
//     thrust::device_vector<cuDoubleComplex> work(lwork);
//     int device_info;
//     // status = cusolverDnZheevd(handle_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
//     //     cols, thrust::raw_pointer_cast(A.data()), cols, 
//     //     thrust::raw_pointer_cast(result.data()), thrust::raw_pointer_cast(work.data()), work.size(), &device_info);
//     // cudaDeviceSynchronize();
//     //if(device_info != 0 || status != 0) { throw std::runtime_error("Failed to run Zheevd"); }
   
//     std::vector<double> host_result(result.size());
//     thrust::copy(result.begin(), result.end(), host_result.begin());
//     cudaDeviceSynchronize();
//     return host_result;
// }

std::vector<double> GpuEigenValSolver::eigenvals(const MatrixType& matrix) {
    assert(matrix.cols() == matrix.rows());
    int cols = matrix.cols();
    thrust::device_vector<float_t> result(cols);

    thrust::host_vector<cuComplex_t> host_A(matrix.size());
    std::transform(matrix.data(), matrix.data() + matrix.size(), host_A.begin(), [](auto v) { return PRECISION_MAKE_FLOAT2(static_cast<float_t>(v.real()), static_cast<float_t>(v.imag())); });
    thrust::device_vector<cuComplex_t> A = host_A;
    cudaDeviceSynchronize();

    // Work size query
    int lwork;
    auto status = PRECISION_HEEVD_bufferSize(handle_, 
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, 
        cols, thrust::raw_pointer_cast(A.data()), cols, thrust::raw_pointer_cast(result.data()), &lwork);
    cudaDeviceSynchronize();
    if(status) { throw std::runtime_error("Failed to get cusolver buffer size"); }

    // Diagonalize
    // TODO: Check storage order and if zgeev is expecting square matrix 
    thrust::device_vector<cuComplex_t> work(lwork);
    thrust::device_vector<int> device_info(1);
    status = PRECISION_HEEVD(handle_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
        cols, thrust::raw_pointer_cast(A.data()), cols, 
        thrust::raw_pointer_cast(result.data()), 
        thrust::raw_pointer_cast(work.data()), lwork, 
        thrust::raw_pointer_cast(device_info.data()));
    cudaDeviceSynchronize();
    if(device_info[0] != 0 || status != 0) { throw std::runtime_error("Failed to run Zheevd"); }
   
    thrust::host_vector<float_t> host_result = result;
    cudaDeviceSynchronize();
    std::vector<double> stl_result(host_result.size());
    std::copy(host_result.begin(), host_result.end(), stl_result.begin());
    return stl_result;
}
}
