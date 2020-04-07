#include "syk.hpp"
#include "util.hpp"
#include "random_matrix.hpp"
#include "filter.hpp"
#include "gauss_hermite.hpp"

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
        output.seekp(-1, std::ios_base::end);
        output << " " << std::endl;
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

template<typename sample_func_t, typename sample_simplex_func_t>
std::vector<double> sample_form_factor(const std::vector<double>& re_time, const sample_func_t& sample_func, const sample_simplex_func_t& sample_simplex, int num_hamiltonians, int avg_count) {
    assert(num_hamiltonians==2);

    std::vector<std::complex<double>> beta(re_time.size());
    std::transform(re_time.begin(), re_time.end(), beta.begin(), [](auto v) { return std::complex<double>(0.0, v); });

    std::vector<double> spectral(beta.size(), 0);

    std::vector<syk::MatrixType> hamiltonian_set(num_hamiltonians);
    #pragma omp parallel
    {
        auto rng_seed = std::random_device()();
        auto rng = std::mt19937_64(rng_seed + omp_get_thread_num());
        util::warmup_rng(&rng);
        
        #pragma omp for
        for(int k = 0; k < hamiltonian_set.size(); ++k) {
            hamiltonian_set[k] = sample_func(&rng);
        }
        
        // This gives some suspect results, but the calculation goes faster??
        // hamiltonian_set[0] = syk::to_eigen_vector(syk::gpu_hamiltonian_eigenvals(hamiltonian_set[0])).asDiagonal();
        
        #pragma omp for
        for(int sample_i = 0; sample_i < util::gauss_hermite_points.size(); ++sample_i) {
            auto x_val = util::gauss_hermite_points[sample_i];
            // Sample Hamiltonian
            double norm = 1.0/std::sqrt(1.0 + x_val*x_val);
            std::vector<double> simplex = {norm, x_val * norm};
            
            auto hamiltonian = util::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), simplex.begin(), 
                syk::MatrixType::Zero(hamiltonian_set[0].rows(), hamiltonian_set[0].cols()).eval());

            // Diagonalize
            auto eigenvals = syk::gpu_hamiltonian_eigenvals(hamiltonian);

            // Spectral form factor
            auto spectral_form_factor = syk::spectral_form_factor(eigenvals);
            std::vector<double> sample_spectral(beta.size());
            std::transform(beta.begin(), beta.end(), sample_spectral.begin(), spectral_form_factor);
            
            #pragma omp critical
            {
                for(int k = 0; k < beta.size(); ++k) {
                    spectral[k] += sample_spectral[k] * util::gauss_hermite_weights[sample_i];
                }
            }
        }
    }

    std::transform(spectral.begin(), spectral.end(), spectral.begin(), [&](const auto& v) { return v; });
    return spectral;
}

int main(int argc, char* argv[]) {
    auto data_path = fs::path("data");
    fs::create_directory(data_path);
    
    std::vector<int> unique_system_sizes {14, 16, 18, 20, 22};
    int avg_count = 10000;
    int trial_count = 1;
    int num_hamiltonians = 2;
    auto re_time = util::logspace(1.0, 1e7-1.0, 10000);
    // auto re_time = util::linspace(1e5, 1e6, 10000);

    std::vector<int> system_sizes(unique_system_sizes.size()*trial_count);
    for(int k = 0; k < unique_system_sizes.size(); ++k) {
        std::fill(system_sizes.begin() + k * trial_count, system_sizes.begin() + (k+1) * trial_count, unique_system_sizes[k]);
    }

    std::vector<std::vector<double>> spectral(system_sizes.size());

    // Spectral form factors
    for(int system_size_i = 0; system_size_i < system_sizes.size(); ++system_size_i){
        int N = system_sizes[system_size_i];

        spectral[system_size_i] = sample_form_factor(
            re_time, 
            [&](auto* rng) { return syk::RandomGUE(rng, 1<<(N/2)); }, 
            [&](auto* rng) { return util::sample_biased_simplex(rng, num_hamiltonians, 0.1); },
            num_hamiltonians,
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
        variance[system_size_i] = std::accumulate(spectral[system_size_i].begin(), spectral[system_size_i].end(), 0.0)/spectral[system_size_i].size();
    }
    
    if(dump_data(data_path / "variance.dat", system_sizes, variance)) { return 1; }
    return 0;
}
