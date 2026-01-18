// SignalTransforms.cpp
// Implementations for SignalTransforms declared in SignalTransforms.h

#include "signal/SignalTransforms.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <boost/math/statistics/univariate_statistics.hpp>
#include <limits>

namespace forge {
    namespace signal {
        std::vector<double> SignalTransforms::tanh_transform(const std::vector<double> &in) {
            std::vector<double> out;
            out.reserve(in.size());
            for (double v: in) out.push_back(std::tanh(v));
            return out;
        }

        void SignalTransforms::tanh_transform_inplace(std::vector<double> &data) {
            for (double &v: data) v = std::tanh(v);
        }

        std::vector<double> SignalTransforms::sigmoid_transform(const std::vector<double> &in) {
            std::vector<double> out;
            out.reserve(in.size());
            for (double v: in) out.push_back(1.0 / (1.0 + std::exp(-v)));
            return out;
        }

        void SignalTransforms::sigmoid_transform_inplace(std::vector<double> &data) {
            for (double &v: data) v = 1.0 / (1.0 + std::exp(-v));
        }

        std::vector<double> SignalTransforms::ranking_transform(const std::vector<double> &in) {
            const std::size_t n = in.size();
            if (n == 0) return {};
            if (n == 1) return {0.0};

            // Pair value with original index
            std::vector<std::pair<double, std::size_t> > v;
            v.reserve(n);
            for (std::size_t i = 0; i < n; ++i) v.emplace_back(in[i], i);

            // Sort by value
            std::sort(v.begin(), v.end(), [](auto const &a, auto const &b) {
                if (a.first < b.first) return true;
                if (a.first > b.first) return false;
                return a.second < b.second;
            });

            // Compute average ranks for ties
            std::vector<double> ranks(n);
            std::size_t i = 0;
            while (i < n) {
                std::size_t j = i + 1;
                while (j < n && v[j].first == v[i].first) ++j;
                double avg_pos = 0.0;
                for (std::size_t k = i; k < j; ++k) avg_pos += static_cast<double>(k);
                avg_pos /= static_cast<double>(j - i);
                for (std::size_t k = i; k < j; ++k) ranks[v[k].second] = avg_pos;
                i = j;
            }

            // Normalize ranks to [0,1] by dividing by (n-1)
            std::vector<double> out(n);
            const double denom = static_cast<double>(n - 1);
            for (std::size_t idx = 0; idx < n; ++idx) out[idx] = ranks[idx] / denom;
            return out;
        }

        void SignalTransforms::ranking_transform_inplace(std::vector<double> &data) {
            data = ranking_transform(data);
        }

        std::vector<double> SignalTransforms::skewness_transform(const std::vector<double> &in, size_t window) {
            const std::size_t n = in.size();
            std::vector<double> skewness(n, std::numeric_limits<double>::quiet_NaN());

            if (window == 0 || n == 0) {
                return skewness; // all NaN
            }

            // If window==1, skewness is not really defined; keep NaNs (or set to 0 if you prefer).
            if (window < 2 || window > n) {
                return skewness;
            }

            // Compute rolling skewness over trailing windows
            for (std::size_t i = window - 1; i < n; ++i) {
                auto first = in.begin() + (i + 1 - window);
                auto last = in.begin() + (i + 1); // one past i
                skewness[i] = boost::math::statistics::skewness(first, last);
            }

            return skewness;
        }

        std::vector<double> SignalTransforms::kurtosis_transform(const std::vector<double> &in, std::size_t window) {
            const std::size_t n = in.size();
            std::vector<double> kurtosis(n, std::numeric_limits<double>::quiet_NaN());

            if (window == 0 || n == 0) {
                return kurtosis; // all NaN
            }

            // If window==1, skewness is not really defined; keep NaNs (or set to 0 if you prefer).
            if (window < 2 || window > n) {
                return kurtosis;
            }

            // Compute rolling skewness over trailing windows
            for (std::size_t i = window - 1; i < n; ++i) {
                auto first = in.begin() + (i + 1 - window);
                auto last = in.begin() + (i + 1); // one past i
                kurtosis[i] = boost::math::statistics::kurtosis(first, last);
            }

            return kurtosis;
        }

        std::vector<double> SignalTransforms::std_transform(const std::vector<double> &in, std::size_t window) {
            const std::size_t n = in.size();
            std::vector<double> stdvector(n, std::numeric_limits<double>::quiet_NaN());

            if (window == 0 || n == 0) {
                return stdvector; // all NaN
            }

            // If window==1, skewness is not really defined; keep NaNs (or set to 0 if you prefer).
            if (window < 2 || window > n) {
                return stdvector;
            }

            // Compute rolling skewness over trailing windows
            for (std::size_t i = window - 1; i < n; ++i) {
                auto first = in.begin() + (i + 1 - window);
                auto last = in.begin() + (i + 1); // one past i
                double s = std::sqrt(boost::math::statistics::sample_variance(first, last));
                stdvector[i] = s;
            }

            return stdvector;
        }
    } // namespace signal
} // namespace forge

