#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

#include "../../libs/signal/include/signal/SignalTransforms.h"

std::vector<double> make_normal_vector(std::size_t n, double mean = 0.0, double stddev = 1.0, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed); // deterministic engine
    std::normal_distribution<double> dist(mean, stddev);

    std::vector<double> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(dist(rng));
    }
    return v;
}

int main() {
    using forge::signal::SignalTransforms;

    const double tol = 1e-12;

    // Test tanh
    std::vector<double> in = {-1.0, 0.0, 1.0};
    auto t = SignalTransforms::tanh_transform(in);
    if (std::fabs(t[0] - std::tanh(-1.0)) > tol) {
        std::cerr << "TANH_FAIL\n";
        return 1;
    }
    if (std::fabs(t[1] - std::tanh(0.0)) > tol) {
        std::cerr << "TANH_FAIL\n";
        return 1;
    }
    if (std::fabs(t[2] - std::tanh(1.0)) > tol) {
        std::cerr << "TANH_FAIL\n";
        return 1;
    }

    // Test sigmoid
    auto s = SignalTransforms::sigmoid_transform(in);
    auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    if (std::fabs(s[0] - sig(-1.0)) > tol) {
        std::cerr << "SIGmoid_FAIL\n";
        return 1;
    }
    if (std::fabs(s[1] - sig(0.0)) > tol) {
        std::cerr << "SIGmoid_FAIL\n";
        return 1;
    }
    if (std::fabs(s[2] - sig(1.0)) > tol) {
        std::cerr << "SIGmoid_FAIL\n";
        return 1;
    }

    // Test ranking with ties
    std::vector<double> r_in = {-1.0, 10.0, 5.0, -31.0};
    auto r = SignalTransforms::ranking_transform(r_in);
    // expected: index0 -> 0/3 = 0.0
    // index1 -> 1/3 = 0.333333333333
    // index2 -> index3 tied -> avg_pos=(2+3)/2=2.5 -> 2.5/3 = 0.833333333333
    if (r.size() != 4) {
        std::cerr << "RANK_SIZE_FAIL\n";
        return 1;
    }
    if (std::fabs(r[0] - 1 / 3.0) > 1e-12) {
        std::cerr << "RANK_FAIL0 " << r[0] << "\n";
        return 1;
    }
    if (std::fabs(r[1] - 1.0) > 1e-12) {
        std::cerr << "RANK_FAIL1 " << r[1] << "\n";
        return 1;
    }
    if (std::fabs(r[2] - (2.0 / 3.0)) > 1e-12) {
        std::cerr << "RANK_FAIL2 " << r[2] << "\n";
        return 1;
    }
    if (std::fabs(r[3] - 0.0) > 1e-12) {
        std::cerr << "RANK_FAIL3 " << r[3] << "\n";
        return 1;
    }

    //test moments:
    std::vector<double> samples = make_normal_vector(5000, 9.5, 150.0);

    size_t window = 500;
    auto kurtosis = SignalTransforms::kurtosis_transform(samples, window); // (mean, stddev, nsamples
    auto minkurtoris = *std::min_element(kurtosis.begin() + window, kurtosis.end());
    auto maxkurtoris = *std::max_element(kurtosis.begin() + window, kurtosis.end());
    if (std::abs(minkurtoris - 3.0) > 1.4 || std::abs(maxkurtoris - 3.0) > 1.4) {
        std::cerr << "KURTOSIS_FAIL\n";
        return 1;
    }

    auto skewness = SignalTransforms::skewness_transform(samples, window); // (mean, stddev, nsamples
    auto minskew = *std::min_element(skewness.begin() + window, skewness.end());
    auto maxskew = *std::max_element(skewness.begin() + window, skewness.end());
    if (std::abs(minskew) > 1 || std::abs(maxskew) > 1) {
        std::cerr << "SKEW_FAIL\n";
        return 1;
    }

    auto std = SignalTransforms::std_transform(samples, window); // (mean, stddev, nsamples
    auto minstd = *std::min_element(std.begin() + window, std.end());
    auto maxsstd = *std::max_element(std.begin() + window, std.end());
    if ((std::abs(minstd) - 150) > 10 || (std::abs(maxsstd) - 150) > 10) {
        std::cerr << "STD_FAIL\n";
        return 1;
    }
    std::cout << "TRANSFORMS_OK" << std::endl;
    return 0;
}

