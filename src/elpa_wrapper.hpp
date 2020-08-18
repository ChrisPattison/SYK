#pragma once
#include "syk_types.hpp"
#include "distributed_matrix.hpp"

#include <string>
#include <cassert>
#include <complex>

#include <mpi.h>

// ELPA is a C library with no C++ support so....
// This is elpa/elpa.h without elpa/elpa_generic.h because _Generic is unsupported
#include <limits.h>
#include <complex.h>

extern "C" {
#define ELPA_H

#include <elpa/elpa_version.h>

struct elpa_struct;
typedef struct elpa_struct *elpa_t;

struct elpa_autotune_struct;
typedef struct elpa_autotune_struct *elpa_autotune_t;

#define complex _Complex
#include <elpa/elpa_constants.h>
#include <elpa/elpa_generated_c_api.h>
#include <elpa/elpa_generated.h>
#undef complex

const char *elpa_strerr(int elpa_error);
}

// Replacing some stuff from elpa_generic...
void elpa_set(elpa_t handle, const char *name, double value, int *error) { elpa_set_double(handle, name, value, error); }
void elpa_set(elpa_t handle, const char *name, int value, int *error) { elpa_set_integer(handle, name, value, error); }

void elpa_get(elpa_t handle, const char *name, double* value, int *error) { elpa_get_double(handle, name, value, error); }
void elpa_get(elpa_t handle, const char *name, int* value, int *error) { elpa_get_integer(handle, name, value, error); }

namespace syk {

// TODO: Look into autotune

class ElpaEigenValSolver {
public:
    void guard_error(int info) {
        if(info != ELPA_OK) {
            std::string error_str(elpa_strerr(info));
            throw std::runtime_error("Error in ELPA call: (" + std::to_string(info) + ") " + error_str);
        }
    }

    ElpaEigenValSolver() {
        int error;
        error = elpa_init(20200417);
        guard_error(error);
    }

    ~ElpaEigenValSolver() {
        int error;
        elpa_uninit(&error);
        // We really shouldn't throw exceptions in constructors but better than leaving a strange state
        guard_error(error);
    }

    ElpaEigenValSolver& operator=(const ElpaEigenValSolver&) = delete;

    std::vector<double> eigenvals(util::distributed_matrix<syk::MatrixType>* matrix, MPI_Comm mpi_comm) {
        assert(matrix->size_x == matrix->size_y);
        assert(matrix->block_size_x == matrix->block_size_y);

        elpa_t handle;
        int error;
        handle = elpa_allocate(&error);
        guard_error(error);

        elpa_set(handle, "na", static_cast<int>(matrix->size_x), &error);
        guard_error(error);
        // TODO: Check column major
        elpa_set(handle, "local_nrows", static_cast<int>(matrix->local_matrix.rows()), &error);
        guard_error(error);
        elpa_set(handle, "local_ncols", static_cast<int>(matrix->local_matrix.cols()), &error);
        guard_error(error);
        elpa_set(handle, "nblk", static_cast<int>(matrix->block_size_x), &error);
        guard_error(error);
        elpa_set(handle, "mpi_comm_parent", static_cast<int>(MPI_Comm_c2f(mpi_comm)), &error);
        guard_error(error);
        elpa_set(handle, "process_col", static_cast<int>(matrix->proc_idx_x), &error);
        guard_error(error);
        elpa_set(handle, "process_row", static_cast<int>(matrix->proc_idx_y), &error);
        guard_error(error);

        error = elpa_setup(handle);
        guard_error(error);

        // elpa_set(handle, "solver", ELPA_SOLVER_2STAGE, &error);
        guard_error(error);

        std::vector<double> eigenvals(matrix->size_y);
        // Double complex and std::complex<double> are usually the same thing
        elpa_eigenvalues_dc(handle, reinterpret_cast<double _Complex*>(matrix->local_matrix.data()), eigenvals.data(), &error);
        guard_error(error);

        elpa_deallocate(handle, &error);
        guard_error(error);
        return eigenvals;
    }
};
}
