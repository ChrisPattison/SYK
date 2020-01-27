#include "syk.hpp"
#include "util.hpp"
#include "random_matrix.hpp"
#include "filter.hpp"

#include <iostream>
#include <exception>
#include <complex>
#include <vector>
#include <random>

using namespace std::complex_literals;

int main(int argc, char* argv[]) {
    std::vector<int> system_sizes {6, 10, 14, 18}; // 2, 6
    int avg_count = 100;
    int num_hamiltonians = 10;
    auto re_time = logspace(1e3, 1e6-1.0, 10000);
    std::vector<std::complex<double>> beta(re_time.size());
    std::transform(re_time.begin(), re_time.end(), beta.begin(), [](auto v) { return std::complex<double>(1.0, v); });
    std::vector<std::vector<double>> spectral(system_sizes.size(), std::vector<double>(beta.size()));

    auto rng = std::mt19937_64(std::random_device()());
    
    // Spectral form factors
    for(int system_size_i = 0; system_size_i < system_sizes.size(); ++system_size_i){
        const auto N = system_sizes.at(system_size_i);

        std::vector<syk::MatrixType> hamiltonian_set(num_hamiltonians);
        // std::generate(hamiltonian_set.begin(), hamiltonian_set.end(), [&]() { return syk::syk_hamiltonian(&rng, N, 1); });
        std::generate(hamiltonian_set.begin(), hamiltonian_set.end(), [&]() { return syk::RandomGUE(&rng, 1<<(N/2)); });

        std::vector<std::vector<double>> simplices(hamiltonian_set.size());
        std::generate(simplices.begin(), simplices.end(), [&]() { return sample_unit_simplex(&rng, hamiltonian_set.size()); });

        std::vector<std::vector<std::complex<double>>> eigenvals(avg_count);

        #pragma omp parallel for
        for(int k = 0; k < num_hamiltonians; ++k) {
            auto hamiltonian = std::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), simplices[k].begin(), 
                syk::MatrixType::Zero(hamiltonian_set[0].rows(), hamiltonian_set[0].cols()).eval());
            eigenvals[k] = syk::hamiltonian_eigenvals(hamiltonian);
        }

        #pragma omp parallel for
        for(int i = 0; i < beta.size(); ++i) {
            spectral[system_size_i][i] = std::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0, std::plus<>(), 
                [&](const auto& v) { return syk::spectral_form_factor(v)(beta[i]); }) / avg_count;
        }
    }

    // Filter step
    #pragma omp parallel for
    for(int system_size_i = 0; system_size_i < spectral.size(); ++system_size_i) {
        auto plataeu = std::accumulate(spectral[system_size_i].begin(), spectral[system_size_i].end(), 0.0)/spectral[system_size_i].size();
        std::transform(spectral[system_size_i].begin(), spectral[system_size_i].end(), spectral[system_size_i].begin(), 
            [&](auto v) { return (v*v - plataeu*plataeu)/(plataeu*plataeu); });
        // spectral[system_size_i] = syk::filter_xy(re_time, spectral[system_size_i], 1e2, 1000);
        #pragma omp critical
        std::cerr << system_sizes[system_size_i] << " " << 
            std::accumulate(spectral[system_size_i].begin(), spectral[system_size_i].end(), 0.0)/spectral[system_size_i].size() << std::endl;
    }

    // Output
    std::cout << "# time" << " ";
    for(int k = 0; k < system_sizes.size(); ++k) {
        std::cout << system_sizes[k] << " ";
    }
    std::cout << std::endl;
    for(int i = 0; i < re_time.size(); ++i) {
        std::cout << re_time.at(i) << " ";
        for(int k = 0; k < system_sizes.size(); ++k) {
            std::cout << spectral.at(k).at(i)/avg_count << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
