// CrossMovingAverage.h
// Header-only utility that uses two ExponentialMovingAverage instances
// to detect crossovers (short crossing long).

#ifndef CURVEFORGE_SIGNAL_CROSSMOVINGAVERAGE_H
#define CURVEFORGE_SIGNAL_CROSSMOVINGAVERAGE_H

#include "ExponentialMovingAverage.h"
#include <optional>

namespace forge {
    namespace signal {
        // CrossMovingAverage: maintains a short and long EMA and returns the numeric
        // difference short - long on each update. Positive value means short > long
        // (short above long), negative means short < long.
        // Usage:
        //   CrossMovingAverage cma(short_period, long_period);
        //   double diff = cma.update(sample); // short - long
        class CrossMovingAverage {
        public:
            // Construct from periods (period -> alpha = 2/(period+1)). periods must be >=1
            CrossMovingAverage(std::size_t short_period, std::size_t long_period);

            // Construct from explicit alphas (0 < alpha <= 1) via named factory to
            // avoid ambiguity with the (size_t, size_t) constructor when integer
            // literals are passed.
            static CrossMovingAverage from_alphas(double short_alpha, double long_alpha) {
                return CrossMovingAverage(short_alpha, long_alpha, /*internal=*/true);
            }

            // Reset both EMAs
            void reset() noexcept {
                short_.reset();
                long_.reset();
                last_relation_.reset();
            }

            // Update with new sample. Returns the numeric difference short - long.
            // The first time both EMAs are seeded this will return the difference
            // between their initialized values.
            double update(double sample);

            // Accessors for current EMA values (may be empty if not seeded)
            std::optional<double> short_value() const noexcept { return short_.value(); }
            std::optional<double> long_value() const noexcept { return long_.value(); }

            // Last computed difference (short - long). Empty until at least one update.
            std::optional<double> last_difference() const noexcept { return last_diff_; }

            // Whether both EMAs have a value
            bool ready() const noexcept { return short_.has_value() && long_.has_value(); }

        private:
            ExponentialMovingAverage short_;
            ExponentialMovingAverage long_;
            std::optional<int> last_relation_; // -1, 0, +1
            std::optional<double> last_diff_;
            // Private constructor used by from_alphas factory
            CrossMovingAverage(double short_alpha, double long_alpha, bool /*internal*/)
                : short_(short_alpha), long_(long_alpha) {
            }
        };
    } // namespace signal
}

#endif // CURVEFORGE_SIGNAL_CROSSMOVINGAVERAGE_H
