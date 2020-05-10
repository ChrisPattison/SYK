#pragma once
#include "syk_types.hpp"
#include <magma.h>
#include <complex>
#include <algorithm>

namespace syk {

class MagmaEigenValSolver {
public:
    MagmaEigenValSolver() { magma_init(); }

    ~MagmaEigenValSolver() { magma_finalize(); }

    MagmaEigenValSolver& operator=(const MagmaEigenValSolver&) = delete;

    std::vector<double> eigenvals(MatrixType& matrix) {
        int size = matrix.diagonal().size();
        int num_eigenvals_found, info;
        std::vector<double> eigenvals(size, std::numeric_limits<double>::quiet_NaN());

        std::vector<magmaDoubleComplex> work(1);
        std::vector<double> rwork(7*size);
        std::vector<magma_int_t> iwork(5*size);

        static_assert(sizeof(std::complex<double>) == sizeof(magmaDoubleComplex));

        magma_zheevx(
            magma_vec_t::MagmaNoVec, magma_range_t::MagmaRangeAll, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaDoubleComplex*>(matrix.data()), matrix.cols(),
            0.0, 0.0,
            -1, -1, -1.0, &num_eigenvals_found,
            eigenvals.data(), 
            nullptr, 1,
            work.data(), -1,
            rwork.data(), iwork.data(),
            nullptr,
            &info);

        work.resize(static_cast<std::size_t>(*reinterpret_cast<double*>(work.data())));

        magma_zheevx(
            magma_vec_t::MagmaNoVec, magma_range_t::MagmaRangeAll, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaDoubleComplex*>(matrix.data()), matrix.cols(),
            0.0, 0.0,
            -1, -1, -1.0, &num_eigenvals_found,
            eigenvals.data(), 
            nullptr, 1,
            work.data(), work.size(),
            rwork.data(), iwork.data(),
            nullptr,
            &info);

        return eigenvals;
    }
};
}
