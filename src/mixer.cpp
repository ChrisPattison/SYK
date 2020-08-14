#include "syk.hpp"
#include "diagonalize.hpp"
#include "magma_diagonalize.hpp"
#include "util.hpp"
#include "boost_util.hpp"
#include "random_matrix.hpp"
#include "filter.hpp"
#include "serdes.hpp"

#include "input_generated.h"
#include "output_generated.h"
#include "checkpoint_generated.h"

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
#include <chrono>
#include <limits>

using namespace std::complex_literals;

namespace fs = std::filesystem;

// TODO: Flatbuffers uses 32 bit offsets -> 2 GiB limit
// TODO: Modify uoffset_t and soffset_t in flatbuffers/base.h at compile time

int main(int argc, char* argv[]) {
    if(argc < 5) {
        std::cerr << "mixer <input_file> <output_file> <samples> <distr_width> [checkpoint_file] [checkpoint_period (s)]" << std::endl;
        return 1;
    }

#if DIAG_SINGLE_PRECISION
    std::cerr << "WARNING: Built with FP32 diagonalization" << std::endl;
#endif

    auto infile_path = fs::path(argv[1]);
    auto outfile_path = fs::path(argv[2]);

    bool checkpoints_enabled = (argc >= 7);

    auto checkpoint_path = checkpoints_enabled ? fs::path(argv[5]) : fs::path();

    // ====== Load Input ======

    unsigned long trials;
    try {
        trials = std::stoul(std::string(argv[3]));
    }catch(const std::invalid_argument&) {
        std::cerr << "Unable to parse number of samples" << std::endl;
        return 1;
    }

    double distr_width;
    try {
        distr_width = std::stod(std::string(argv[4]));
    }catch(const std::invalid_argument&) {
        std::cerr << "Unable to parse distribution width" << std::endl;
        return 1;
    }

    unsigned long checkpoint_period = std::numeric_limits<unsigned long>::max();
    if(checkpoints_enabled) {
        try {
            checkpoint_period = std::stoul(std::string(argv[6]));
        }catch(const std::invalid_argument&) {
            std::cerr << "Unable to parse checkpoint period" << std::endl;
            return 1;
        }
    }


    std::list<syk::MatrixType> hamiltonian_set;
    std::array<std::uint32_t, 5> input_hash;
    try {
        // Buffer size
        auto input_file = H5::H5File(infile_path.c_str(), H5F_ACC_RDONLY);
        if(!input_file.nameExists("hamiltonian")) {
            throw std::runtime_error("/hamiltonian missing in HDF5 input file");
        }

        auto group = input_file.openGroup("hamiltonian");
        auto num_attributes = group.getNumObjs();

        // Load matrices
        for(unsigned int k = 0; k < num_attributes; ++k) {
            hamiltonian_set.push_back(util::load_matrix_hdf5(&group, group.getObjnameByIdx(k)));
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
    std::size_t output_buffer_reservation = trials * hamiltonian_set.front().rows() * sizeof(syk::MatrixType::Scalar);
    flatbuffers::FlatBufferBuilder output_builder(output_buffer_reservation);
    std::vector<flatbuffers::Offset<SYKSchema::Point>> points;
    points.reserve(trials);
    std::int64_t total_compute = 0;

    // Load checkpoint if available
    if(checkpoints_enabled && fs::exists(checkpoint_path)) {
        try{
            auto checkpoint_file = std::fstream(checkpoint_path, std::ios_base::in | std::ios_base::binary);
            auto num_bytes = util::get_stream_size(&checkpoint_file);
            // Read buffer
            std::vector<char> buf(num_bytes);
            checkpoint_file.read(buf.data(), num_bytes);
            if(checkpoint_file.fail()) { throw std::runtime_error("Unable to open file"); }

            auto checkpoint_schema = SYKSchema::GetCheckpoint(buf.data());
            if(checkpoint_schema->input_hash() == nullptr) { throw std::runtime_error("Missing input file hash field in checkpoint"); }
            decltype(input_hash) checkpoint_input_hash;
            std::copy(checkpoint_schema->input_hash()->data()->cbegin(), checkpoint_schema->input_hash()->data()->cend(), checkpoint_input_hash.begin());
            if(checkpoint_input_hash != input_hash) { throw std::runtime_error("Input file changed from checkpoint"); }

            if(checkpoint_schema->output() == nullptr || checkpoint_schema->output()->data() == nullptr) { throw std::runtime_error("Missing output or output data field in checkpoint"); }

            points = util::copy_points(*checkpoint_schema->output()->data(), &output_builder);
            total_compute = checkpoint_schema->output()->total_compute();
        }catch(const std::exception& e) {
            std::cerr << "Failed to load from checkpoint: " << e.what() << std::endl;
            return 1;
        }
    }

    auto last_checkpoint = std::chrono::steady_clock::now();
    // Compute
    {
        auto rng_seed = std::random_device()();
        auto rng = std::mt19937_64(rng_seed + omp_get_thread_num());
        util::warmup_rng(&rng);

        std::cerr << "Using CPUs" << std::endl;

        // #pragma omp single
        #pragma omp parallel for
        for(int sample_i = points.size(); sample_i < trials; ++sample_i) {
            #pragma omp critical
            std::cerr << "Sample #: " << sample_i << std::endl;
            
            // Sample Hamiltonian
            std::vector<double> x_vals(hamiltonian_set.size());

            #pragma omp critical
            std::generate(x_vals.begin()+1, x_vals.end(), [&]() { return std::normal_distribution(0.0, 1.0)(rng) * distr_width; });
            x_vals[0] = 1;

            double sum_squares = util::transform_reduce(x_vals.begin()+1, x_vals.end(), 0.0, std::plus<>(), [](const auto& v) { return v * v;} );
            double norm = 1.0;///std::sqrt(1.0 + sum_squares);
            std::transform(x_vals.begin(), x_vals.end(), x_vals.begin(), [&](const auto& v) { return v * norm;});

            syk::MatrixType hamiltonian = util::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), x_vals.begin(), 
                syk::MatrixType::Zero(hamiltonian_set.front().rows(), hamiltonian_set.front().cols()).eval());

            // Diagonalize
            std::vector<double> eigenvals;
            try {
                eigenvals = syk::cpu_hamiltonian_eigenvals(hamiltonian);
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
        total_compute += std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - last_checkpoint).count();
    }

    // ====== Dump Output ======
    
    auto output = SYKSchema::CreateOutput(output_builder, total_compute, output_builder.CreateVector(points));
    output_builder.Finish(output);
    output_file.write(reinterpret_cast<char*>(output_builder.GetBufferPointer()), output_builder.GetSize());
    if(output_file.fail()) {
        std::cerr << "Failed to write output to disk" << std::endl;
        return 1;
    }

    return 0;
}
