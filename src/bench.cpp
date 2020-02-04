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
				auto eigenvals = syk::hamiltonian_eigenvals(hamiltonian);
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
