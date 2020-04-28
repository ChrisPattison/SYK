#pragma once
#include "input_generated.h"
#include "output_generated.h"
#include "syk_types.hpp"
#include "util.hpp"

#include <complex>

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
}