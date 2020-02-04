#include "diagonalize.hpp"

#include <cassert>
#include <algorithm>
#include <complex>
#include <exception>

#include <thrust/copy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


namespace syk {

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
    thrust::device_vector<float> result(cols);

    thrust::host_vector<cuFloatComplex> host_A(matrix.size());
    std::transform(matrix.data(), matrix.data() + matrix.size(), host_A.begin(), [](auto v) { return make_float2(static_cast<float>(v.real()), static_cast<float>(v.imag())); });
    thrust::device_vector<cuFloatComplex> A = host_A;
    cudaDeviceSynchronize();

    // Work size query
    int lwork;
    auto status = cusolverDnCheevd_bufferSize(handle_, 
        CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, 
        cols, thrust::raw_pointer_cast(A.data()), cols, thrust::raw_pointer_cast(result.data()), &lwork);
    cudaDeviceSynchronize();
    if(status) { throw std::runtime_error("Failed to get cusolver buffer size"); }

    // Diagonalize
    // TODO: Check storage order and if zgeev is expecting square matrix 
    thrust::device_vector<cuFloatComplex> work(lwork);
    thrust::device_vector<int> device_info(1);
    status = cusolverDnCheevd(handle_, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
        cols, thrust::raw_pointer_cast(A.data()), cols, 
        thrust::raw_pointer_cast(result.data()), 
        thrust::raw_pointer_cast(work.data()), lwork, 
        thrust::raw_pointer_cast(device_info.data()));
    cudaDeviceSynchronize();
    if(device_info[0] != 0 || status != 0) { throw std::runtime_error("Failed to run Zheevd"); }
   
    thrust::host_vector<float> host_result = result;
    cudaDeviceSynchronize();
    std::vector<double> stl_result(host_result.size());
    std::copy(host_result.begin(), host_result.end(), stl_result.begin());
    return stl_result;
}
}
