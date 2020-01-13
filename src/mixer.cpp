#include "syk.hpp"
#include "util.hpp"

#include <iostream>
#include <exception>
#include <complex>
#include <vector>
#include <random>

using namespace std::complex_literals;

int main(int argc, char* argv[]) {
    int N = 16;
    int avg_count = 100;
    int num_hamiltonians = 10;
    auto beta = logspace(1e-1i, 1e6i, 10000);
    std::transform(beta.begin(), beta.end(), beta.begin(), [](auto v) { return v + static_cast<std::complex<double>>(1.0); });

    auto rng = std::mt19937_64(std::random_device()());
    
    std::vector<syk::MatrixType> hamiltonian_set(num_hamiltonians);
    std::generate(hamiltonian_set.begin(), hamiltonian_set.end(), [&]() { return syk::syk_hamiltonian(&rng, N, 1); });

    std::vector<std::vector<double>> simplices(hamiltonian_set.size());
    std::generate(simplices.begin(), simplices.end(), [&]() { return sample_unit_simplex(&rng, hamiltonian_set.size()); });

    std::vector<std::vector<std::complex<double>>> eigenvals(avg_count);

    #pragma omp parallel for static
    for(int k = 0; k < num_hamiltonians; ++k) {
        auto hamiltonian = std::transform_reduce(hamiltonian_set.begin(), hamiltonian_set.end(), simplices[k].begin(), 
            syk::MatrixType::Zero(hamiltonian_set[0].rows(), hamiltonian_set[0].cols()).eval());
        eigenvals[k] = syk::hamiltonian_eigenvals(hamiltonian);
    }

    std::vector<double> spectral(beta.size(), 0);
    #pragma omp parallel for static
    for(int i = 0; i < beta.size(); ++i) {
        spectral[i] = std::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0, std::plus<>(), 
            [&](const auto& v) { return syk::spectral_form_factor(v)(beta[i]); }) / avg_count;
    }
    
    for(int i = 0; i < beta.size(); ++i) {
        std::cout << beta[i].imag() << " " << spectral[i]/avg_count << std::endl;
    }
    return 0;
}
