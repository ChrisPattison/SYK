#pragma once

#include <cusolverDn.h>
#include <vector>

#include "syk_types.hpp"

namespace syk {

class GpuEigenValSolver {
    cusolverDnHandle_t handle_;
public:
    GpuEigenValSolver();

    ~GpuEigenValSolver();

    GpuEigenValSolver(const GpuEigenValSolver&) : GpuEigenValSolver() { }

    GpuEigenValSolver& operator=(const GpuEigenValSolver&) = delete;

    std::vector<double> eigenvals(const MatrixType& matrix);
};
}
