#pragma once
#include <algorithm>
#include <numeric>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <omp.h>

namespace syk_plot {
template<typename func_t, typename add_func_t, typename acc_t, typename zero_func_t>
auto lazy_pairwise_sum(func_t gen, add_func_t add_func, acc_t sum, long long lower, long long upper, zero_func_t zero_func) -> acc_t {
    int serial_size = 1024;

    if(upper-lower < serial_size) {
        for(long long k = lower; k < upper; ++k) {
            gen(&sum, k);
        }
    }else {
        auto split = (upper-lower)/2;
        auto a = lazy_pairwise_sum(gen, add_func, zero_func(), lower, lower + split, zero_func);
        auto b = lazy_pairwise_sum(gen, add_func, zero_func(), lower + split, upper, zero_func);
        sum = add_func(sum, a);
        sum = add_func(sum, b);
    }
    return sum;
}

void spectral_form_factor_impl(double* t, long long t_len, double* a, long long a_len, double* output, long long output_len) {
    assert(output_len == t_len);

    std::vector<std::complex<double>> spectral(output_len);
    #pragma omp parallel
    {
        int thread_count = omp_get_num_threads();
        int thread_num = omp_get_thread_num();
        long long chunck_size = (a_len + thread_count - 1) / thread_count;
        long long chunk_start = thread_num * chunck_size;

        auto add_func = 
            [&](const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b) {
                std::vector<std::complex<double>> c(a.size());
                for(long long i = 0; i < c.size(); ++i) { c[i] = a[i] + b[i]; }
                return c;
            };
        auto gen_func = [&](std::vector<std::complex<double>>* sum, int k) {
                #pragma omp simd
                for(long long i = 0; i < t_len; ++i) { (*sum)[i] += std::complex<double>(std::cos(t[i] * a[k]), std::sin(t[i] * a[k])); }
            };

        auto zero_func = [&]() { return std::vector<std::complex<double>>(t_len); };
        auto accumulate = lazy_pairwise_sum(gen_func, add_func, zero_func(), chunk_start, std::min(chunk_start + chunck_size, a_len), zero_func);

        #pragma omp critical
        for(int i = 0; i < spectral.size(); ++i) {
            spectral[i] += accumulate[i];
        }
    }

    std::transform(spectral.begin(), spectral.end(), output, [&](const auto& z) { return std::norm(z) / static_cast<double>(a_len * a_len); });
}
}

