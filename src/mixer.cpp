#include "syk.hpp"
#include "util.hpp"
#include "random_matrix.hpp"
#include "filter.hpp"

#include <iostream>
#include <fstream>
#include <exception>
#include <complex>
#include <vector>
#include <random>
#include <string>
#include <cassert>
#include <filesystem>

using namespace std::complex_literals;

namespace fs = std::filesystem;

template<typename system_size_t, typename time_t, typename spectral_t>
bool dump_data(fs::path file_path, const std::vector<system_size_t>& system_sizes, 
    const std::vector<time_t>& re_time, const std::vector<spectral_t>& spectral) {
        
    auto output = std::fstream(file_path, std::ios_base::out | std::ios_base::trunc );
    if(output.fail()) { return true; }

    // Output
    output << "time" << ",";
    for(int k = 0; k < system_sizes.size(); ++k) {
        output << system_sizes.at(k) << ",";
    }
    output << std::endl;
    for(int i = 0; i < re_time.size(); ++i) {
        output << re_time.at(i) << ",";
        for(int k = 0; k < system_sizes.size(); ++k) {
            output << spectral.at(k).at(i) << ",";
        }
        output << std::endl;
    }
    return false;
}

template<typename system_size_t, typename value_t>
bool dump_data(fs::path file_path, const std::vector<system_size_t>& system_sizes, const std::vector<value_t>& values) {
        
    auto output = std::fstream(file_path, std::ios_base::out | std::ios_base::trunc );
    if(output.fail()) { return true; }

    // Output
    for(int k = 0; k < system_sizes.size(); ++k) {
        output << system_sizes.at(k) << "," << values.at(k) << std::endl;
    }
    return false;
}

template<typename sample_func_t, typename sample_symplex_func_t>
std::vector<double> sample_form_factor(const std::vector<double>& re_time, const sample_func_t& sample_func, const sample_symplex_func_t& sample_symplex, int avg_count) {

    std::vector<std::complex<double>> beta(re_time.size());
    std::transform(re_time.begin(), re_time.end(), beta.begin(), [](auto v) { return std::complex<double>(1.0, v); });

    std::vector<double> spectral(beta.size(), 0);

    std::vector<syk::MatrixType> hamiltonian_set(sample_symplex().size());
    std::generate(hamiltonian_set.begin(), hamiltonian_set.end(), sample_func);

    for(int sample_i = 0; sample_i < avg_count; ++sample_i) {
        // Sample Hamiltonian
        auto symplex = sample_symplex();
        auto hamiltonian = std::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), symplex.begin(), 
            syk::MatrixType::Zero(hamiltonian_set[0].rows(), hamiltonian_set[0].cols()).eval());

        // Diagonalize
        auto eigenvals = syk::hamiltonian_eigenvals(hamiltonian);

        // Spectral form factor
        auto spectral_form_factor = syk::spectral_form_factor(eigenvals);
        #pragma omp parallel for
        for(int i = 0; i < beta.size(); ++i) {
            spectral[i] += spectral_form_factor(beta[i]); 
        }
    }

    std::transform(spectral.begin(), spectral.end(), spectral.begin(), [&](const auto& v) { return v / avg_count; });
    return spectral;
}

int main(int argc, char* argv[]) {
    auto data_path = fs::path("data");
    fs::create_directory(data_path);
    
    std::vector<int> unique_system_sizes {6, 10, 14, 18};
    // std::vector<int> unique_system_sizes {18};
    int avg_count = 1000;
    int trial_count = 1;
    int num_hamiltonians = 2;
    // auto re_time = logspace(1e5, 1e7-1.0, 100000);
    auto re_time = logspace(10.0, 1e7-1.0, 100000);

    std::vector<int> system_sizes(unique_system_sizes.size()*trial_count);
    for(int k = 0; k < unique_system_sizes.size(); ++k) {
        std::fill(system_sizes.begin() + k * trial_count, system_sizes.begin() + (k+1) * trial_count, unique_system_sizes[k]);
    }
    auto rng = std::mt19937_64(std::random_device()());
    std::vector<std::vector<double>> spectral(system_sizes.size());
    
    // Spectral form factors
    for(int system_size_i = 0; system_size_i < system_sizes.size(); ++system_size_i){
        int N = system_sizes[system_size_i];

        spectral[system_size_i] = sample_form_factor(
            re_time, 
            // [&]() { return syk::RandomGUE(&rng, 1<<(N/2)); }, 
            [&]() { return syk::syk_hamiltonian(&rng, N, 1); },
            [&]() { return sample_biased_simplex(&rng, num_hamiltonians, 0.1); },
            avg_count);
    }

    if(dump_data(data_path / "plataeu.dat", system_sizes, re_time, spectral)) { return 1; }

    // Filter step
    std::vector<double> variance(system_sizes.size());
    #pragma omp parallel for
    for(int system_size_i = 0; system_size_i < spectral.size(); ++system_size_i) {
        auto plataeu = std::accumulate(spectral[system_size_i].begin(), spectral[system_size_i].end(), 0.0)/spectral[system_size_i].size();
        std::transform(spectral[system_size_i].begin(), spectral[system_size_i].end(), spectral[system_size_i].begin(), 
            [&](auto v) { return (v*v - plataeu*plataeu)/(plataeu*plataeu); });
        // spectral[system_size_i] = syk::filter_xy(re_time, spectral[system_size_i], 1e2, 1000);
        #pragma omp critical
        variance[system_size_i] = std::accumulate(spectral[system_size_i].begin(), spectral[system_size_i].end(), 0.0)/spectral[system_size_i].size();
    }
    
    if(dump_data(data_path / "variance.dat", system_sizes, variance)) { return 1; }

    if(dump_data(data_path / "filtered.dat", system_sizes, re_time, spectral)) { return 1; }
    return 0;
}
