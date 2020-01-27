#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <cassert>

namespace syk {

std::vector<double> gaussian_window(double window_width, double step_size, double cutoff) {
    double distance = -window_width * std::log(cutoff);
    int int_steps = static_cast<int>(std::ceil(distance/step_size));
    std::vector<double> window(2*int_steps+1);
    auto midpoint = window.begin() + int_steps;
    {
        int n = 0;
        std::generate(midpoint, window.begin(), [&]() { 
            return std::exp(-std::pow(step_size*n++, 2)/ (window_width * window_width)); });
        std::copy(midpoint, window.end(), std::make_reverse_iterator(midpoint));
    }
    double norm = std::accumulate(window.begin(), window.end(), 0);
    std::transform(window.begin(), window.end(), window.begin(), [&](auto& v) { return v / norm; });
    return window;
}

template<typename data_type>
auto filter_vec(const std::vector<data_type>& data, const std::vector<double>& window) {
    assert(window.size() % 2 == 1);
    return [&](const int& k) {
        data_type v = 0;
        int half_width = (window.size() - 1)/2;
        auto start_index = std::max(0, k - half_width);
        auto stop_index = std::min(data.size(), k + half_width + 1);
        assert(stop_index - start_index <= window.size());

        for(int i = start_index; i < stop_index; ++i) {
            v += window[i] * data;
        }
        return v;
    };
}

template<typename x_type, typename y_type>
std::vector<double> filter_xy(const std::vector<x_type> x, const std::vector<y_type> y, x_type width, int window_cutoff_distance) {
    assert(x.size() == y.size());
    std::vector<y_type> y_filtered(y.size());
    for(int k = 0; k < y_filtered.size(); ++k) {
        int window_start = std::max(0, k - window_cutoff_distance);
        int window_stop = std::min(static_cast<int>(x.size())-1, k + window_cutoff_distance);
        y_type val = 0;
        y_type norm = 0;
        auto weight_func = [&](const x_type v) { return std::exp(-(v*v) / (width*width)); };
        for(int i = window_start; i < window_stop; ++i) {
            auto dx = x[i+1] - x[i];
            auto midpoint = (x[i+1] + x[i])/2;
            norm += dx * weight_func(midpoint);
            val += dx * weight_func(midpoint) * (y[i+1] + y[i])/2;
        }
        y_filtered[k] = val/norm;
    }
    return y_filtered;
}
}