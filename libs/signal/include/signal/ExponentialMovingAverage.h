// ExponentialMovingAverage.h
// Header-only utility for computing an exponential moving average (EMA).
// Provides a small, readable API: update(), value(), has_value(), reset().

#ifndef CURVEFORGE_SIGNAL_EXPONENTIALMOVINGAVERAGE_H
#define CURVEFORGE_SIGNAL_EXPONENTIALMOVINGAVERAGE_H

#include <optional>

namespace forge {
    namespace signal {
        // Simple exponential moving average (EMA) calculator.
        // Formula: EMA_n = alpha * x_n + (1 - alpha) * EMA_{n-1}
        // alpha must be in (0, 1]. You can construct from a smoothing factor alpha
        // or from a period using `from_period(period)` which uses alpha = 2/(period+1).
        class ExponentialMovingAverage {
        public:
            // Construct with explicit smoothing factor alpha (0 < alpha <= 1)
            explicit ExponentialMovingAverage(double alpha) : alpha_(alpha) {
            };

            // Construct from an integer period (common in finance/time-series):
            // alpha = 2 / (period + 1). period must be >= 1.
            static ExponentialMovingAverage from_period(std::size_t period);

            // Reset to empty (no value). The next update will seed the EMA with the first sample.
            void reset() noexcept { value_.reset(); }

            // Reset to a specific initial value.
            void reset(double initial) noexcept { value_ = initial; }

            // Update the EMA with a new sample and return the updated EMA value.
            // If the EMA has no prior value, the first sample initializes it.
            double update(double sample);

            // Convenience: operator() does the same as update().
            double operator()(double sample) { return update(sample); }

            // Return the current EMA value if available.
            std::optional<double> value() const noexcept { return value_; }

            // Whether the EMA has been seeded with at least one sample.
            bool has_value() const noexcept { return value_.has_value(); }

            // Return the configured alpha (smoothing factor).
            double alpha() const noexcept { return alpha_; }

        private:
            double alpha_;
            std::optional<double> value_;
        };
    } // namespace forge::signal
}
#endif // CURVEFORGE_SIGNAL_EXPONENTIALMOVINGAVERAGE_H

