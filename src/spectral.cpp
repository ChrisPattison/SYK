/* Copyright (c) 2020 C. Pattison
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
#include "syk.hpp"
#include "diagonalize.hpp"
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
