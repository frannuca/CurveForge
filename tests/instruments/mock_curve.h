#pragma once

#include <chrono>
#include "curve/ICurve.h"
#include "time/daycount.hpp"


// A simple mock curve for testing that returns a constant value.
class MockCurve : public curve::ICurve {
public:
    explicit MockCurve(double constant_rate)
        : curve::ICurve(
              curve::time::Date{std::chrono::year{2025} / 11 / 16},
              {
                  {{std::chrono::year{2025} / 11 / 16}, 0.02},
                  {{std::chrono::year{2099} / 11 / 16}, 0.02},
              }, curve::time::create_daycount_convention(curve::time::DayCountConvention::ACT_365F)
          ), constant_rate_(constant_rate) {
    }

    [[nodiscard]] std::string name() const override { return "MockCurve"; }

private:
    double constant_rate_;
};

