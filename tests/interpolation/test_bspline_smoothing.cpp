#include "interpolation/bspline.h"
#include <Eigen/Dense>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace interpolation;

// Compute a simple roughness metric: sum of squared second finite differences of y component
static double roughness(const std::vector<double> &xs, const std::vector<double> &ys) {
    double r = 0.0;
    if (ys.size() < 3) return 0.0;
    for (size_t i = 2; i < ys.size(); ++i) {
        double d2 = ys[i] - 2 * ys[i - 1] + ys[i - 2];
        r += d2 * d2;
    }
    return r;
}

int main() {
    const int N = 25; // data points
    std::vector<Eigen::VectorXd> noisy;
    noisy.reserve(N);
    for (int i = 0; i < N; ++i) {
        double x = double(i) / (N - 1);
        double y = x * x * x;
        double noise = 0.05 * std::rand() / double(RAND_MAX); // deterministic noise
        if (i != 0 && i != N - 1) y += noise;
        Eigen::Vector2d pt;
        pt << x, y;
        noisy.push_back(pt);
    }

    auto exact = bspline::interpolate(noisy, 3, "uniform");

    // Sample exact
    const int M = 101;
    std::vector<double> xs;
    xs.reserve(M);
    std::vector<double> yExact;
    yExact.reserve(M);
    for (int i = 0; i < M; ++i) {
        double u = double(i) / (M - 1);
        auto ve = exact->evaluate(u);
        xs.push_back(u);
        yExact.push_back(ve[1]);
    }
    double rExact = roughness(xs, yExact);

    std::vector<double> lambdas{0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    bool passed = false;
    for (double lambda: lambdas) {
        auto smooth = bspline::smooth_interpolate(noisy, 3, lambda, "uniform");
        std::vector<double> ySmooth;
        ySmooth.reserve(M);
        for (int i = 0; i < M; ++i) {
            double u = xs[i];
            auto vs = smooth->evaluate(u);
            ySmooth.push_back(vs[1]);
        }
        double rSmooth = roughness(xs, ySmooth);
        if (rSmooth < rExact * 0.85) {
            // at least 15% reduction
            passed = true;
            break;
        }
    }

    assert(passed && "Smoothing failed to reduce roughness by expected margin");
    std::cout << "SMOOTH_OK\n";
    return 0;
}
