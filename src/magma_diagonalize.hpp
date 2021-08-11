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
#include "syk_types.hpp"
#include <magma.h>
#include <complex>
#include <algorithm>
#include <type_traits>
#include <string>

namespace syk {

class MagmaEigenValSolver {
#if DIAG_SINGLE_PRECISION
    using float_t = float;
    using magmaComplex_t = magmaFloatComplex;
    static constexpr auto& heevd_m = magma_cheevd_m;
    static constexpr auto& heevdx_2stage_m = magma_cheevdx_2stage_m;
#else
    using float_t = double;
    using magmaComplex_t = magmaDoubleComplex;
    static constexpr auto& heevd_m = magma_zheevd_m;
    static constexpr auto& heevdx_2stage_m = magma_zheevdx_2stage_m;
#endif

    auto mat_to_working_precision(MatrixType* matrix) {
        if constexpr (std::is_same_v<MatrixType::Scalar::value_type, std::complex<float_t>>) {
            return std::move(*matrix);
        }else {
            Eigen::Matrix<std::complex<float_t>, Eigen::Dynamic, Eigen::Dynamic> new_matrix = matrix->cast<std::complex<float_t>>();
            return new_matrix;
        }
    }

    auto vector_from_working_precision(std::vector<float_t>* a) {
        if constexpr (std::is_same_v<float_t, double>) {
            return std::move(*a);
        }else {
            std::vector<double> converted(a->size());
            std::copy(a->begin(), a->end(), converted.begin());
            return converted;
        }
    }

public:
    MagmaEigenValSolver() { magma_init(); }

    ~MagmaEigenValSolver() { magma_finalize(); }

    MagmaEigenValSolver& operator=(const MagmaEigenValSolver&) = delete;

    void guard_error(magma_int_t info) {
        if(info != MAGMA_SUCCESS) { 
            std::string err = "Magma error " + std::to_string(info);
            throw std::runtime_error(err + ". " + magma_strerror(info));
        }
    }

    std::vector<double> eigenvals(MatrixType& matrix) {
        magma_int_t ngpu = num_gpus();
        magma_int_t size = matrix.diagonal().size();
        magma_int_t num_eigenvals_found, info;
        std::vector<float_t> eigenvals(size, std::numeric_limits<float_t>::quiet_NaN());

        std::vector<magmaComplex_t> work(1);
        std::vector<float_t> rwork(1);
        std::vector<magma_int_t> iwork(1);

        static_assert(sizeof(std::complex<float_t>) == sizeof(magmaComplex_t));

        auto A_mat = mat_to_working_precision(&matrix);

        // Work query
        heevd_m(ngpu,
            magma_vec_t::MagmaNoVec, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaComplex_t*>(A_mat.data()), A_mat.cols(),
            eigenvals.data(), 
            work.data(), -1,
            rwork.data(), -1,
            iwork.data(), -1,
            &info);
        guard_error(info);

        work.resize(static_cast<std::size_t>(*reinterpret_cast<float_t*>(work.data())));
        rwork.resize(static_cast<std::size_t>(rwork[0]));
        iwork.resize(iwork[0]);

        // Diagonalization call
        heevd_m(ngpu,
            magma_vec_t::MagmaNoVec, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaComplex_t*>(A_mat.data()), A_mat.cols(),
            eigenvals.data(), 
            work.data(), work.size(),
            rwork.data(), rwork.size(),
            iwork.data(), iwork.size(),
            &info);
        guard_error(info);

        return vector_from_working_precision(&eigenvals);
    }

    std::vector<double> eigenvals_2stage(MatrixType& matrix) {
        magma_int_t ngpu = num_gpus();
        magma_int_t size = matrix.diagonal().size();
        magma_int_t num_eigenvals_found, info;
        std::vector<float_t> eigenvals(size, std::numeric_limits<float_t>::quiet_NaN());

        std::vector<magmaComplex_t> work(1);
        std::vector<float_t> rwork(1);
        std::vector<magma_int_t> iwork(1);

        static_assert(sizeof(std::complex<float_t>) == sizeof(magmaComplex_t));

        auto A_mat = mat_to_working_precision(&matrix);

        // Work query
        heevdx_2stage_m(ngpu,
            magma_vec_t::MagmaNoVec, magma_range_t::MagmaRangeAll, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaComplex_t*>(A_mat.data()), A_mat.cols(),
            std::numeric_limits<float_t>::quiet_NaN(), std::numeric_limits<float_t>::quiet_NaN(), 0, 0,
            &num_eigenvals_found,
            eigenvals.data(), 
            work.data(), -1,
            rwork.data(), -1,
            iwork.data(), -1,
            &info);
        guard_error(info);

        work.resize(static_cast<std::size_t>(*reinterpret_cast<float_t*>(work.data())));
        rwork.resize(static_cast<std::size_t>(rwork[0]));
        iwork.resize(iwork[0]);

        // Diagonalization call
        heevdx_2stage_m(ngpu,
            magma_vec_t::MagmaNoVec, magma_range_t::MagmaRangeAll, magma_uplo_t::MagmaLower, size, 
            reinterpret_cast<magmaComplex_t*>(A_mat.data()), A_mat.cols(),
            std::numeric_limits<float_t>::quiet_NaN(), std::numeric_limits<float_t>::quiet_NaN(), 0, 0,
            &num_eigenvals_found,
            eigenvals.data(), 
            work.data(), work.size(),
            rwork.data(), rwork.size(),
            iwork.data(), iwork.size(),
            &info);
        guard_error(info);

        return vector_from_working_precision(&eigenvals);
    }

    int num_gpus() {
        return static_cast<int>(magma_num_gpus());
    }
};
}
