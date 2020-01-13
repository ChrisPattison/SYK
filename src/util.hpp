#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <cassert>

template<typename T>
std::vector<T> logspace(T a, T b, int N) {
    int n = 0;
    std::vector<T> vals(N);
    auto m = (std::log(b) - std::log(a)) / static_cast<T>(N);
    std::generate(vals.begin(), vals.end(), [&]() {
        return std::exp(m * static_cast<T>(n++) + std::log(a)); });
    return vals;
}

// http://blog.geomblog.org/2005/10/sampling-from-simplex.html
template<typename rng_type>
std::vector<double> sample_unit_simplex(rng_type* rand_gen, int dim) {
    assert(dim >= 2);
    std::uniform_real_distribution distr;
    std::vector<double> vals(dim+2);
    std::generate(vals.begin()+2, vals.end(), [&]() { return distr(*rand_gen); });
    vals[0] = 0;
    vals[1] = 1;
    std::sort(vals.begin(), vals.end());
    std::adjacent_difference(vals.begin(), vals.end(), vals.begin());
    vals.erase(vals.begin());
    assert(std::abs(std::accumulate(vals.begin(), vals.end(), 0.0) - 1.0) < 1e-7);
    return vals;
}