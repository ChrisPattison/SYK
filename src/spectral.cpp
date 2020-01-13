#include "syk.hpp"
#include "util.hpp"

#include <iostream>
#include <exception>
#include <complex>
#include <vector>
#include <random>

using namespace std::complex_literals;

int main(int argc, char* argv[]) {
    int N = 14;
    int avg_count = 1000;
    auto beta = logspace(1e-1i, 1e6i, 10000);
    std::transform(beta.begin(), beta.end(), beta.begin(), [](auto v) { return v + static_cast<std::complex<double>>(1.0); });

    auto rng = std::mt19937_64(std::random_device()());
    
    std::vector<std::vector<std::complex<double>>> eigenvals(avg_count);
    for(int k = 0; k < avg_count; ++k) {
        auto hamiltonian = syk::syk_hamiltonian(&rng, N, 1);
        eigenvals[k] = syk::hamiltonian_eigenvals(hamiltonian);
    }

    std::vector<double> spectral(beta.size(), 0);
    for(int i = 0; i < beta.size(); ++i) {
        spectral[i] = std::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0, std::plus<>(), 
            [&](const auto& v) { return syk::spectral_form_factor(v)(beta[i]); }) / avg_count;
    }
    
    for(int i = 0; i < beta.size(); ++i) {
        std::cout << beta[i].imag() << " " << spectral[i]/avg_count << std::endl;
    }
    return 0;
}
