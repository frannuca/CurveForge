#include "pricing/curve/yield_curve.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pricing {
    YieldCurve::YieldCurve() { nodes_.push_back({0.0, 1.0}); }

    void YieldCurve::add(double t, double df) {
        if (t <= 0.0) throw std::invalid_argument("Node time must be >0");
        for (auto &n: nodes_) if (std::fabs(n.t - t) < 1e-12) throw std::invalid_argument("Duplicate node");
        if (!(df > 0.0)) throw std::invalid_argument("Discount must be positive");
        nodes_.push_back({t, df});
        std::sort(nodes_.begin(), nodes_.end(), [](auto &a, auto &b) { return a.t < b.t; });
    }


    double YieldCurve::discount(double t) const {
        if (t == 0.0) return 1.0;
        for (auto &n: nodes_) if (std::fabs(n.t - t) < 1e-12) return n.df;
        if (t < 0.0 || t > nodes_.back().t) throw std::out_of_range("Extrapolation not allowed");
        for (size_t i = 1; i < nodes_.size(); ++i)
            if (t < nodes_[i].t) {
                auto &a = nodes_[i - 1];
                auto &b = nodes_[i];
                double w = (t - a.t) / (b.t - a.t);
                double lnDF = (1 - w) * std::log(a.df) + w * std::log(b.df);
                return std::exp(lnDF);
            }
        return nodes_.back().df;
    }
} // namespace pricing
