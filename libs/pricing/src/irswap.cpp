#include "pricing/curve/irswap.hpp"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <cmath>

namespace pricing {
    IRSwap::IRSwap(std::vector<double> payTimes, double fixedRate)
        : payTimes_(std::move(payTimes)), fixedRate_(fixedRate) {
        if (payTimes_.empty()) throw std::invalid_argument("IRSwap needs payments");
        for (size_t i = 1; i < payTimes_.size(); ++i)
            if (payTimes_[i] <= payTimes_[i - 1]) throw std::invalid_argument("IRSwap times not ascending");
    }

    double IRSwap::maturity() const { return payTimes_.back(); }
    const char *IRSwap::type() const { return "SWAP"; }
    const std::vector<double> &IRSwap::paymentTimes() const { return payTimes_; }

    double IRSwap::solveDiscount(const std::function<double(double)> &discount) const {
        // Bootstrapping objective: given par fixed rate k and earlier discount factors D(t_i) for i < n,
        // solve D(T_n) from: k * sum_{i=1..n} alpha_i * D(t_i) = 1 - D(T_n)
        // => D(T_n) * (1 + k * alpha_n) = 1 - k * sum_{i=1..n-1} alpha_i * D(t_i)
        // => D(T_n) = (1 - k * sumKnown) / (1 + k * alpha_n)
        const double eps = 1e-14;
        size_t n = payTimes_.size();
        if (n == 0) throw std::logic_error("IRSwap empty payTimes after construction");

        if (n == 1) { // Degenerate single-period swap: treat like deposit
            double T = payTimes_.back();
            if (!(T > 0.0)) throw std::invalid_argument("Invalid single payment maturity");
            double alpha = T; // daycount assumed Act/1 basis externally
            double denom = 1.0 + fixedRate_ * alpha;
            if (denom <= eps) throw std::runtime_error("Denominator non-positive in single-period swap");
            return 1.0 / denom;
        }

        double sumKnown = 0.0;
        double prev = 0.0;
        // Accumulate fixed leg PV excluding last period
        for (size_t i = 0; i + 1 < n; ++i) {
            double t = payTimes_[i];
            if (!(t > prev)) throw std::runtime_error("Non-increasing payment time encountered");
            double alpha = t - prev;
            double df = discount(t); // should succeed (earlier node or interpolated)
            if (!(df > 0.0)) throw std::runtime_error("Non-positive discount factor returned for earlier payment");
            sumKnown += alpha * df;
            prev = t;
        }

        double Tn = payTimes_.back();
        if (!(Tn > prev)) throw std::runtime_error("Final maturity not greater than previous payment");
        double alphaN = Tn - prev;

        double numerator = 1.0 - fixedRate_ * sumKnown;
        double denom = 1.0 + fixedRate_ * alphaN;

        if (denom <= eps) throw std::runtime_error("Denominator non-positive solving final discount factor");
        double df = numerator / denom;

        // Basic sanity checks
        if (!(df > 0.0)) throw std::runtime_error("Solved non-positive DF for IRSwap");
        if (df > 1.0 + 1e-10) throw std::runtime_error("Solved discount factor > 1 (inconsistent inputs)");
        return df;
    }
} // namespace pricing
