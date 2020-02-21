#include "syk.hpp"
#include "util.hpp"
#include "random_matrix.hpp"

#include <iostream>
#include <exception>
#include <complex>
#include <vector>
#include <random>

#include <omp.h>

using namespace std::complex_literals;

int main(int argc, char* argv[]) {
    int N = 22;
    int avg_count = 1000;
    auto beta = util::logspace(1e-0i, 1e6i, 10000);
    std::transform(beta.begin(), beta.end(), beta.begin(), [](auto v) { return v + static_cast<std::complex<double>>(0.0); });

    auto rng_seed = std::random_device()();
    
    
    std::vector<std::vector<double>> eigenvals(avg_count);
    #pragma omp parallel
    {
        auto rng = std::mt19937_64(rng_seed + omp_get_thread_num());
        util::warmup_rng(&rng);

        #pragma omp for
        for(int k = 0; k < avg_count; ++k) {
            syk::MatrixType hamiltonian;
            hamiltonian = syk::RandomGUE(&rng, 1<<(N/2));
            eigenvals[k] = syk::gpu_hamiltonian_eigenvals(hamiltonian);
        }
    }

    std::vector<double> spectral(beta.size(), 0);
    for(int i = 0; i < beta.size(); ++i) {
        spectral[i] = util::transform_reduce(eigenvals.cbegin(), eigenvals.cend(), 0.0, std::plus<>(), 
            [&](const auto& v) { return syk::spectral_form_factor(v)(beta[i]); }) / avg_count;
    }
    
    std::cout << "beta, " << N << std::endl;
    for(int i = 0; i < beta.size(); ++i) {
        std::cout << beta[i].imag() << ", " << spectral[i]/avg_count << std::endl;
    }
    return 0;
}
