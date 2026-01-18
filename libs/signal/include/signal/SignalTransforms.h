// SignalTransforms.h
// Header-only utilities for transforming numeric vectors (tanh, sigmoid, ranking)

#ifndef CURVEFORGE_SIGNAL_TRANSFORMS_H
#define CURVEFORGE_SIGNAL_TRANSFORMS_H

#include <vector>

namespace forge {
    namespace signal {
        class SignalTransforms {
        public:
            // Return a new vector with tanh applied element-wise
            static std::vector<double> tanh_transform(const std::vector<double> &in);

            // In-place tanh
            static void tanh_transform_inplace(std::vector<double> &data);

            // Sigmoid (logistic) function 1/(1+exp(-x)) element-wise
            static std::vector<double> sigmoid_transform(const std::vector<double> &in);

            static void sigmoid_transform_inplace(std::vector<double> &data);

            // Ranking: map values to [0,1] according to their rank (0 -> smallest, 1 -> largest)
            // Handles ties by assigning the average rank to tied values (fractional rank), then
            // normalizing by (n-1). For n==1 returns {0.0}.
            static std::vector<double> ranking_transform(const std::vector<double> &in);

            static void ranking_transform_inplace(std::vector<double> &data);

            static std::vector<double> skewness_transform(const std::vector<double> &in, size_t window);

            static std::vector<double> kurtosis_transform(const std::vector<double> &in, size_t window);

            static std::vector<double> std_transform(const std::vector<double> &in, size_t window);
        };
    } // namespace signal
} // namespace forge

#endif // CURVEFORGE_SIGNAL_TRANSFORMS_H
