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
 
#include "random_matrix.hpp"
#include "syk.hpp"
#include "diagonalize.hpp"
#include <iostream>
#include <exception>
#include <chrono>
#include <complex>
#include <random>

using namespace std::complex_literals;

int main(int argc, char* argv[]) {
	std::complex<double> sum = 0;
	std::vector<int> N_vals {6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
	int bench_count = 10;

	auto rng = std::mt19937_64(std::random_device()());
	
	for(int N : N_vals) {
		try {
			std::chrono::microseconds time_elapsed(0);
			
			auto time_start = std::chrono::high_resolution_clock::now();
			for (int k = 0; k < bench_count; ++k) {
				// auto hamiltonian = syk::syk_hamiltonian(&rng, N, 1); // This has cubic time complexity???
				auto hamiltonian = syk::RandomGUE(&rng, 1<<(N/2));
				auto eigenvals = syk::gpu_hamiltonian_eigenvals(hamiltonian);
				sum += std::accumulate(eigenvals.begin(), eigenvals.end(), 0.0i);
			}
			time_elapsed += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - time_start);

			std::cerr << N << "  "  << time_elapsed.count()/bench_count << std::endl;
		} catch(std::runtime_error& ex) {
			std::cerr << ex.what() << std::endl;
			return 1;
		}
	}
	return 0;
}
