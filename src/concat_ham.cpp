#include "input_generated.h"
#include "serdes.hpp"

#include <iostream>
#include <fstream>
#include <exception>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void print_args() {
    std::cerr
        << "concat_ham <hamiltonian_1> <hamiltonian_2> ... <hamiltonian_n> <destination_file>" << std::endl;
}

int main(int argc, char* argv[]) {

    // Argument parsing
    if(argc < 3) {
        print_args();
        return 1;
    }

    std::vector<fs::path> inputs;
    for(int k = 1; k < argc-1; ++k) {
        inputs.push_back(fs::path(argv[k]));

        // if(!fs::exists(inputs.back)) {
        //     std::cerr << "Missing input file " << inputs.back() << std::endl;
        //     return 1;
        // }

        if(!fs::is_regular_file(inputs.back())) {
            std::cerr << "Input argument " << inputs.back() << " is not a file" << std::endl;
            return 1;
        }
    }

    auto destination = fs::path(argv[argc-1]);
    if(fs::exists(destination)) {
        std::cerr << "Destination file " << destination << " exists" << std::endl;
        return 1;
    }

    // Check output file writable

    std::fstream output_file;
    try {
        output_file = std::fstream(destination, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc | std::ios_base::ate);
        output_file.put(0);
        output_file.flush();
        output_file.seekg(0, std::ios::beg);
        if(output_file.fail()) { throw std::runtime_error("Unable to write to output file"); }
    }catch(const std::exception& e) {
        std::cerr << "Output file writability check failed: " << e.what() << std::endl;
        return 1;
    }

    // Grab hamiltonians

    std::vector<syk::MatrixType> matrices;
    for(auto& input_path : inputs) {
        // Buffer size
        auto input_file = std::fstream(input_path, std::ios_base::in | std::ios_base::binary);
        auto num_bytes = util::get_stream_size(&input_file);

        // Read buffer
        std::vector<char> buf(num_bytes);
        input_file.read(buf.data(), num_bytes);
        if(input_file.fail()) { 
            std::cerr << "Unable to read from " << input_path << std::endl;
            return 1;
        }

        auto input_schema = SYKSchema::GetInput(buf.data());

        // Hamiltonian nullptr check
        if(input_schema->hamiltonians() == nullptr) {
            std::cerr << "Missing Hamiltonians in " << input_path << std::endl;
        }

        // Load matrices
        for(auto matrix : *(input_schema->hamiltonians())) {
            matrices.push_back(util::load_matrix(*matrix));
        }
    }

    // Serialize and write output

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
