//
// B-Spline Smoothing Demonstration
// Generates data comparing exact vs smooth interpolation for documentation
//

#include "interpolation/bspline.h"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

int main() {
    // Generate noisy data from a smooth function
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> noise(0.0, 0.02);

    std::vector<Eigen::VectorXd> noisy_data;

    std::cout << "# Data for B-Spline Smoothing Comparison\n";
    std::cout << "# Format: x, true_value, noisy_value, exact_interp, smooth_interp\n\n";

    // Generate 20 sample points with noise
    const int n_samples = 20;
    for (int i = 0; i < n_samples; ++i) {
        double x = i / (n_samples - 1.0);
        double true_y = x * x * x - 0.5 * x * x + 0.1 * x; // Cubic function
        double noisy_y = true_y + noise(gen);

        Eigen::Vector2d point;
        point << x, noisy_y;
        noisy_data.push_back(point);
    }

    // Create exact interpolation (passes through all noisy points)
    auto exact_spline = interpolation::bspline::interpolate(noisy_data, 3, "chord");

    // Create smooth interpolation (reduces noise)
    auto smooth_spline = interpolation::bspline::smooth_interpolate(
        noisy_data, 3, 0.1, "chord"
    );

    // Evaluate and output results at many points for plotting
    const int n_eval = 100;
    for (int i = 0; i <= n_eval; ++i) {
        double u = i / double(n_eval);
        // Evaluate splines
        auto exact_val = exact_spline->evaluate(u);
        auto smooth_val = smooth_spline->evaluate(u);

        double x = exact_val[0];
        double true_y = x * x * x - 0.5 * x * x + 0.1 * x;

        std::cout << std::fixed << std::setprecision(6)
                << x << ","
                << true_y << ","
                << exact_val[1] << ","
                << smooth_val[1] << "\n";
    }

    // Also output the noisy sample points
    std::cout << "\n# Noisy sample points:\n";
    for (const auto &pt: noisy_data) {
        std::cout << std::fixed << std::setprecision(6)
                << pt[0] << "," << pt[1] << "\n";
    }

    return 0;
}

