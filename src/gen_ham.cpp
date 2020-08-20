#include "syk.hpp"
#include "heisenberg.hpp"
#include "random_matrix.hpp"
#include "input_generated.h"
#include "serdes.hpp"

#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <random>
#include <cassert>
#include <filesystem>

#include <H5Cpp.h>

namespace fs = std::filesystem;

enum class HamType { syk, gue, goe, heis };

void print_args() {
    std::cerr
        << "gen_ham <output_file> <num_hamiltonians> <ensemble [syk/gue/goe/heis]> <size>" << std::endl
        << "Restrictions on size:" << std::endl
        << "SYK: # of fermions" << std::endl
        << "GUE/GOE: Vector space dimension" << std::endl
        << "Random Field Heisenberg: # of qubits" << std::endl;
}

int main(int argc, char* argv[]) {
    if(argc < 5) {
        print_args();
        return 1;
    }

    // ===== Load Args =======
    auto outfile_path = fs::path(argv[1]);
    int num_hamiltonians, size;
    try {
        num_hamiltonians = std::stoi(std::string(argv[2]));
        size = std::stoi(std::string(argv[4]));
        if(num_hamiltonians < 1) { throw std::runtime_error("Non-positive hamiltonian count"); }
        if(size < 1) { throw std::runtime_error("Non-positive size"); }
    }catch(const std::exception& e) {
        print_args();
        std::cerr << "Failed to parse arguments: " << e.what() << std::endl;
        return 1;
    }

    auto ensemble_arg = std::string(argv[3]);
    HamType ham_type;
    if      (ensemble_arg == "syk")  { ham_type = HamType::syk; }
    else if (ensemble_arg == "gue")  { ham_type = HamType::gue; }
    else if (ensemble_arg == "goe")  { ham_type = HamType::goe; }
    else if (ensemble_arg == "heis") { ham_type = HamType::heis; }
    else {
        print_args();
        std::cerr << "Invalid ensemble type. Valid types are <syk/gue/goe/heis>" << std::endl;
        return 1;
    }

    // ===== Open output file =======

    H5::H5File output_file;
    try {
        output_file = H5::H5File(outfile_path.c_str(), H5F_ACC_TRUNC);
    }catch(const std::exception& e) {
        std::cerr << "Open output file failed: " << e.what() << std::endl;
        return 1;
    }

    auto ham_group = output_file.createGroup("hamiltonian");

    // ====== Gen Data ======

    auto rng = std::mt19937_64(std::random_device()());
    util::warmup_rng(&rng);

    try {
        for(int k = 0 ; k < num_hamiltonians; ++k) {
            syk::MatrixType matrix;
            switch(ham_type) {
                case HamType::gue:  matrix = syk::RandomGUE(&rng, size); break;
                case HamType::goe:  matrix = syk::RandomGOE(&rng, size); break;
                case HamType::syk:  matrix = syk::syk_hamiltonian(&rng, size, 1.0); break;
                case HamType::heis: matrix = syk::random_field_heisenberg(&rng, size, 0.5); break;
            }
            auto ham_name = std::string("ham_") + std::to_string(k);
            util::dump_matrix_hdf5(matrix, &ham_group, ham_name);
        }

    }catch(const std::exception& e) {
        std::cerr << "Failed to generate or write hamiltonians: " << e.what() << std::endl;
    }

    return 0;
}
