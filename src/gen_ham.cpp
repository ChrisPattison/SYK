#include "syk.hpp"
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

namespace fs = std::filesystem;

enum class HamType { syk, gue, goe };

void print_args() {
    std::cerr
        << "gen_ham <output_file> <num_hamiltonians> <ensemble [syk/gue/goe]> <size>" << std::endl
        << "Restrictions on size:" << std::endl
        << "SYK: # of fermions" << std::endl
        << "GUE/GOE: Vector space dimension" << std::endl;
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
    if      (ensemble_arg == "syk") { ham_type = HamType::syk; }
    else if (ensemble_arg == "gue") { ham_type = HamType::gue; }
    else if (ensemble_arg == "goe") { ham_type = HamType::goe; }
    else {
        print_args();
        std::cerr << "Invalid ensemble type. Valid types are <syk/gue/goe>" << std::endl;
        return 1;
    }

    // ===== Open output file =======

    std::fstream output_file;
    try {
        output_file = std::fstream(outfile_path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc | std::ios_base::ate);
        output_file.put(0);
        output_file.flush();
        output_file.seekg(0, std::ios::beg);
        if(output_file.fail()) { throw std::runtime_error("Unable to write to output file"); }
    }catch(const std::exception& e) {
        std::cerr << "Output file writability check failed: " << e.what() << std::endl;
        return 1;
    }

    // ====== Gen Data ======

    std::vector<syk::MatrixType> matrices(num_hamiltonians);

    auto rng = std::mt19937_64(std::random_device()());
    util::warmup_rng(&rng);

    switch(ham_type) {
        case HamType::gue: std::generate(matrices.begin(), matrices.end(), [&]() { return syk::RandomGUE(&rng, size); }); break;
        case HamType::goe: std::generate(matrices.begin(), matrices.end(), [&]() { return syk::RandomGOE(&rng, size); }); break;
        case HamType::syk: std::generate(matrices.begin(), matrices.end(), [&]() { return syk::syk_hamiltonian(&rng, size, 1.0); }); break;
    }

    // ====== Output Data =====

    flatbuffers::FlatBufferBuilder builder(matrices.size() * matrices.front().size() * sizeof(syk::MatrixType::Scalar));
    std::vector<flatbuffers::Offset<SYKSchema::Matrix>> matrix_offsets(matrices.size());

    std::transform(matrices.begin(), matrices.end(), matrix_offsets.begin(), [&](const auto& mat) { return util::dump_matrix(mat, &builder); });
    auto output = SYKSchema::CreateInput(builder, builder.CreateVector(matrix_offsets));
    builder.Finish(output);

    output_file.write(reinterpret_cast<char*>(builder.GetBufferPointer()), builder.GetSize());
    if(output_file.fail()) {
        std::cerr << "Failed to write output to disk" << std::endl;
        return 1;
    }

    return 0;
}