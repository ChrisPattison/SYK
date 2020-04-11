#include "syk.hpp"
#include "diagonalize.hpp"
#include "util.hpp"
#include "random_matrix.hpp"
#include "filter.hpp"
#include "serdes.hpp"

#include "input_generated.h"
#include "output_generated.h"

#include <iostream>
#include <fstream>
#include <exception>
#include <complex>
#include <vector>
#include <list>
#include <random>
#include <string>
#include <cassert>
#include <filesystem>

using namespace std::complex_literals;

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if(argc < 5) {
        std::cerr << "mixer <input_file> <output_file> <samples> <distr_width>" << std::endl;
        return 1;
    }

    auto infile_path = fs::path(argv[1]);
    auto outfile_path = fs::path(argv[2]);

    // ====== Load Input ======

    int trials;
    try {
        trials = std::stoi(std::string(argv[3]));
    }catch(const std::invalid_argument&) {
        std::cerr << "Unable to parse number of samples" << std::endl;
        return 1;
    }
    if(trials <= 0) {
        std::cerr << "Invalid number of samples" << std::endl;
        return 1;
    }

    double distr_width;
    try {
        distr_width = std::stod(std::string(argv[4]));
    }catch(const std::invalid_argument&) {
        std::cerr << "Unable to parse distribution width" << std::endl;
        return 1;
    }


    std::list<syk::MatrixType> hamiltonian_set;
    try {
        // Buffer size
        auto input_file = std::fstream(infile_path, std::ios_base::in | std::ios_base::binary);
        input_file.seekg(0, std::ios::end);
        std::size_t num_bytes = input_file.tellg();
        input_file.seekg(0, std::ios::beg);

        // Read buffer
        std::vector<char> buf(num_bytes);
        input_file.read(buf.data(), num_bytes);
        if(input_file.fail()) { throw std::runtime_error("Unable to open file"); }

        auto input_schema = SYKSchema::GetInput(buf.data());
        
        // Hamiltonian nullptr check
        if(input_schema->hamiltonians() == nullptr) { throw std::runtime_error("Missing Hamiltonians in input file"); }

        // Load matrices
        for(auto matrix : *(input_schema->hamiltonians())) {
            hamiltonian_set.push_back(util::load_matrix(*matrix));
        }
        // Check number
        if(hamiltonian_set.size() == 0) { throw std::runtime_error("Found zero Hamiltonians"); }

        // Check sizes
        for(const auto& v : hamiltonian_set) { 
            if(v.cols() != hamiltonian_set.front().cols() || v.rows() != hamiltonian_set.front().rows())
                { throw std::runtime_error("Hamiltonian size mismatch"); }
        }
    }catch(const std::exception& e) {
        std::cerr << "Failed to parse input file: " << e.what() << std::endl;
        return 1;
    }

    // Check output writable
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

    // ====== Compute ======

    flatbuffers::FlatBufferBuilder output_builder(trials * hamiltonian_set.front().rows() * sizeof(syk::MatrixType::Scalar));
    std::vector<flatbuffers::Offset<SYKSchema::Point>> points;
    points.reserve(trials);
    {
        auto rng_seed = std::random_device()();
        auto rng = std::mt19937_64(rng_seed + omp_get_thread_num());
        util::warmup_rng(&rng);

        for(int sample_i = 0; sample_i < trials; ++sample_i) {
            // Sample Hamiltonian
            std::vector<double> x_vals(hamiltonian_set.size());
            std::generate(x_vals.begin()+1, x_vals.end(), [&]() { return std::normal_distribution(0.0, 1.0)(rng) * distr_width; });
            x_vals[0] = 1;

            double sum_squares = util::transform_reduce(x_vals.begin()+1, x_vals.end(), 0.0, std::plus<>(), [](const auto& v) { return v * v;} );
            double norm = 1.0/std::sqrt(1.0 + sum_squares);
            std::transform(x_vals.begin(), x_vals.end(), x_vals.begin(), [&](const auto& v) { return v * norm;});

            syk::MatrixType hamiltonian = util::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), x_vals.begin(), 
                syk::MatrixType::Zero(hamiltonian_set.front().rows(), hamiltonian_set.front().cols()).eval());

            // Diagonalize
            std::vector<double> eigenvals;
            try {
                eigenvals = syk::gpu_hamiltonian_eigenvals(hamiltonian);
            }catch(const std::exception& e) {
                std::cerr << "Failure during diagonalization: " << e.what() << std::endl;
                throw;
            }
            
            // Save data
            #pragma omp critical
            {
                auto params_vector = output_builder.CreateVector(x_vals);
                auto eigenval_vector = output_builder.CreateVector(eigenvals);
                auto point = SYKSchema::CreatePoint(output_builder, params_vector, eigenval_vector);
                points.push_back(point);
            }
        }
    }

    // ====== Dump Output ======
    
    auto output = SYKSchema::CreateOutput(output_builder, output_builder.CreateVector(points));
    output_builder.Finish(output);
    output_file.write(reinterpret_cast<char*>(output_builder.GetBufferPointer()), output_builder.GetSize());
    if(output_file.fail()) {
        std::cerr << "Failed to write output to disk" << std::endl;
        return 1;
    }

    return 0;
}
