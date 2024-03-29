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
#include "input_generated.h"
#include "output_generated.h"
#include "syk_types.hpp"
#include "util.hpp"
#include "distributed_matrix.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <type_traits>
#include <complex>
#include <exception>
#include <string>
#include <iostream>

namespace util {
static_assert(!(syk::MatrixType::IsRowMajor));

/** Load matrix from FlatBuffers schema
 */
syk::MatrixType load_matrix(const SYKSchema::Matrix& serialized) {
    if(serialized.data() == nullptr) { throw std::runtime_error("No matrix data"); }
    if(serialized.size_n() * serialized.size_m() != serialized.data()->size()) { throw std::runtime_error("Matrix size does not match"); }

    syk::MatrixType matrix(serialized.size_n(), serialized.size_m());
    for(int j = 0; j < matrix.rows(); ++j) {
        for(int i = 0; i < matrix.cols(); ++i) {
            const auto& coeff = serialized.data()->Get(i + j*matrix.rows());
            assert(coeff != nullptr); // Flatbuffers will never return a nullptr from Get
            matrix(i,j) = std::complex<double> {coeff->real(), coeff->imag()};
        }
    }

    return matrix;
}

/** Serialize matrix into FlatBuffers schema
 */
flatbuffers::Offset<SYKSchema::Matrix> dump_matrix(const syk::MatrixType& matrix, flatbuffers::FlatBufferBuilder* builder) {
    std::function serialize = [&](std::size_t i, SYKSchema::Complex64* dest) { 
            const auto& src = matrix.data()[i]; 
            *dest = SYKSchema::Complex64(src.real(), src.imag());
        };
    auto vec = builder->CreateVectorOfStructs(static_cast<std::size_t>(matrix.size()), serialize);
    return SYKSchema::CreateMatrix(*builder, matrix.rows(), matrix.cols(), vec);
}

/** Copy serialized output data into fresh buffer
 */
template<typename points_container>
std::vector<flatbuffers::Offset<SYKSchema::Point>> copy_points(const points_container& points, flatbuffers::FlatBufferBuilder* builder) {
    std::vector<flatbuffers::Offset<SYKSchema::Point>> dest_points;
    
    for(const auto& src_point : points) {
        if(src_point->params() == nullptr || src_point->eigenvals() == nullptr)
            { std::runtime_error("Missing field in checkpoint data point"); }

        std::function copy_params = [&](std::size_t index) {
            const auto& val_ptr = src_point->params()->Get(index++);
            assert(val_ptr != nullptr);
            return val_ptr;
        };
        auto params_vector = builder->CreateVector(src_point->params()->size(), copy_params);
        
        std::function copy_eigenvals = [&](std::size_t index) {
            const auto& val_ptr = src_point->eigenvals()->Get(index++);
            assert(val_ptr != nullptr);
            return val_ptr;
        };
        auto eigenvals_vector = builder->CreateVector(src_point->eigenvals()->size(), copy_eigenvals);


        auto output_point = SYKSchema::CreatePoint(*builder, params_vector, eigenvals_vector);
        dest_points.push_back(output_point);
    }
    return dest_points;
}

/** Load Eigen matrix from HDF5 Group
 */
syk::MatrixType load_matrix_hdf5(H5::Group* group, std::string name) {
    static_assert(std::is_same_v<syk::MatrixType::Scalar, std::complex<double> >);
    static_assert(sizeof(std::complex<double>) == 16);

    auto dataset = group->openDataSet(name);

    // Read matrix size
    auto dataspace = dataset.getSpace();
    if(dataspace.getSimpleExtentNdims() != 3) {
        throw std::runtime_error("Datset rank mismatch. Expected 3");
    }

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims);
    if(dims[2] != 2) {
        throw std::runtime_error("Dataset trailing dimension not 2 (Complex)");
    }
    
    // Read matrix
    // Opposite ordering of HDF5
    syk::MatrixType matrix(dims[1], dims[0]);
    dataset.read(reinterpret_cast<double*>(matrix.data()), H5::PredType::NATIVE_DOUBLE);

    return matrix;
}

/** Utility for computing dimenions of processor grid
 * Requires that number of processors has a prime factorization in terms of 2, 3, 5, and 7
 */
std::pair<std::size_t, std::size_t> get_proc_grid(std::size_t num_procs) {
    std::array<std::size_t, 4> factors {2, 3, 5, 7};
    std::array<std::size_t, 4> factor_count {0, 0, 0, 0};
    auto total_procs = num_procs;

    for(int k = 0; k < factors.size(); ++k) {
        // Divisibility check and some sane bounds just in case 
        while(1048576 > num_procs && num_procs > 0 && num_procs%factors[k] == 0) {
            num_procs /= factors[k];
            factor_count[k]++;
        }
    }
    if(num_procs != 1) {
        throw std::runtime_error("Unable to find prime factorization of processor count. Processor count is restricted to integer multiples of {2, 3, 5, 7}.");
    }

    std::pair<std::size_t, std::size_t> grid {1,1};
    // Greedy pairing
    for(int k = factors.size()-1; k >= 0; --k) {
        while(factor_count[k] > 0) {
            if(grid.first < grid.second) {
                grid.first *= factors[k];
            }else {
                grid.second *= factors[k];
            }
            factor_count[k]--;
        }
    }
    
    if(grid.first * grid.second != total_procs) {
        throw std::runtime_error("Product of processor grid dimensions (" + std::to_string(grid.first) + ", " + std::to_string(grid.second) + ")"
            + "does not equal total number of processors " + std::to_string(total_procs) + " . (This is a bug.)");
    }
    return grid;
}

/** Load part of a block cyclic matrix into memory from an HDF5 achive
 */
distributed_matrix<syk::MatrixType> load_block_cyclic_matrix_hdf5(H5::Group* group, std::string name, 
    std::size_t num_procs, std::size_t proc_idx, std::pair<std::size_t, std::size_t> block_size) {

    static_assert(std::is_same_v<syk::MatrixType::Scalar, std::complex<double> >);
    static_assert(sizeof(std::complex<double>) == 16);

    auto proc_grid = get_proc_grid(num_procs);

    auto dataset = group->openDataSet(name);

    // Read matrix size
    auto dataspace = dataset.getSpace();
    if(dataspace.getSimpleExtentNdims() != 3) {
        throw std::runtime_error("Datset rank mismatch. Expected 3");
    }

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims);
    if(dims[2] != 2) {
        throw std::runtime_error("Dataset trailing dimension not 2 (Complex)");
    }


    distributed_matrix<syk::MatrixType> matrix;
    matrix.size_x = dims[0];
    matrix.size_y = dims[1];

    matrix.num_procs_x = proc_grid.first;
    matrix.num_procs_y = proc_grid.second;

    matrix.proc_idx_x = proc_idx % matrix.num_procs_x;
    matrix.proc_idx_y = (proc_idx / matrix.num_procs_x) % matrix.num_procs_y;

    matrix.block_size_x = block_size.first;
    matrix.block_size_y = block_size.second;


    // This will be "backwards" because eigen <-> hdf5 row/col ordering
    std::int64_t destination_cols;
    std::int64_t destination_rows;

    dataspace.selectNone();
    // Full blocks
    hsize_t offset[3] = {matrix.proc_idx_x * matrix.num_procs_x, matrix.proc_idx_y * matrix.num_procs_y, 0};
    hsize_t block[3] = {matrix.block_size_x, matrix.block_size_y, 2};
    hsize_t stride[3] = {matrix.block_size_x * matrix.num_procs_x, matrix.block_size_y * matrix.num_procs_y, 1};
    hsize_t count[3] = {dims[0] / (matrix.block_size_x * matrix.num_procs_x), dims[1] / (matrix.block_size_y * matrix.num_procs_y), 1};
    dataspace.selectHyperslab(H5S_SELECT_OR, count, offset, stride, block);

    destination_cols += block[0] * count[0];
    destination_rows += block[1] * count[1];

    // TODO: HDF5 will blow up if excess is 0
    // TODO: Check if we get 0s anywhere to account for 0 in eigenval distribution bug
    // TODO: Accidently transposed?
    // First Excess
    {
        hsize_t excess_offset[3] = {stride[0] * count[0], matrix.proc_idx_y * matrix.num_procs_y, 0};
        hsize_t excess_block[3] = {
            std::min(static_cast<std::size_t>((dims[0] >= excess_offset[0]) ? dims[0] - excess_offset[0] : 0), matrix.block_size_x),
            matrix.block_size_y,
            2};
        hsize_t excess_stride[3] = {1, matrix.block_size_y * matrix.num_procs_y, 1};
        hsize_t excess_count[3] = {1, dims[1] / (matrix.block_size_y * matrix.num_procs_y), 1};
        if(excess_block[0] > 0) {
            dataspace.selectHyperslab(H5S_SELECT_OR, excess_count, excess_offset, excess_stride, excess_block);
        }
    }

    // Second Excess
    {
        hsize_t excess_offset[3] = {matrix.proc_idx_x * matrix.num_procs_x, stride[1] * count[1], 0};
        hsize_t excess_block[3] = {
            matrix.block_size_x, 
            std::min(static_cast<std::size_t>((dims[1] >= excess_offset[1]) ? dims[1] - excess_offset[1] : 0), matrix.block_size_y), 
            2};
        hsize_t excess_stride[3] = {matrix.block_size_x * matrix.num_procs_x, 1, 1};
        hsize_t excess_count[3] = {dims[0] / (matrix.block_size_x * matrix.num_procs_x), 1, 1};
        if(excess_block[1] > 0) {
            dataspace.selectHyperslab(H5S_SELECT_OR, excess_count, excess_offset, excess_stride, excess_block);
        }
    }

    // First and Second Excess
    {
        hsize_t excess_offset[3] = {stride[0] * count[0], stride[1] * count[1], 0};
        hsize_t excess_block[3] = {
            std::min(static_cast<std::size_t>((dims[0] >= excess_offset[0]) ? dims[0] - excess_offset[0] : 0), matrix.block_size_x),
            std::min(static_cast<std::size_t>((dims[1] >= excess_offset[1]) ? dims[1] - excess_offset[1] : 0), matrix.block_size_y),
            2};
        hsize_t excess_stride[3] = {1, 1, 1};
        hsize_t excess_count[3] = {1, 1, 1};

        if(excess_block[0] > 0 && excess_block[1] > 0) {
            dataspace.selectHyperslab(H5S_SELECT_OR, excess_count, excess_offset, excess_stride, excess_block);

            destination_cols += excess_block[0] * excess_count[0];
            destination_rows += excess_block[1] * excess_count[1];
        }
    }

    matrix.local_matrix.resize(destination_rows, destination_cols);
    hsize_t destination_dims[3] = {static_cast<hsize_t>(matrix.local_matrix.cols()), static_cast<hsize_t>(matrix.local_matrix.rows()), 2};
    dataset.read(reinterpret_cast<double*>(matrix.local_matrix.data()), H5::PredType::NATIVE_DOUBLE, H5::DataSpace(3, destination_dims), dataspace);

    return matrix;
}

/** Store Eigen matrix into HDF5 Group
 */
void dump_matrix_hdf5(const syk::MatrixType& matrix, H5::Group* group, std::string name) {
    static_assert(std::is_same_v<syk::MatrixType::Scalar, std::complex<double> >);
    static_assert(sizeof(std::complex<double>) == 16);

    // We're actually storing the transpose here since HDF5 is row-major and Eigen is col-major
    hsize_t dataset_dims[3] =  {static_cast<hsize_t>(matrix.cols()), static_cast<hsize_t>(matrix.rows()), 2};
    auto dataset = group->createDataSet(name, H5::PredType::INTEL_F64, H5::DataSpace(3, dataset_dims));
    dataset.write(reinterpret_cast<const double*>(matrix.data()), H5::PredType::NATIVE_DOUBLE);
}
}
