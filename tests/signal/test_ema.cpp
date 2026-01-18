#include <iostream>
#include <cmath>
#include "signal/ExponentialMovingAverage.h"


int main() {
    using namespace forge::signal;

    // Create EMA with period 3 -> alpha = 2/(3+1) = 0.5
    ExponentialMovingAverage ema = ExponentialMovingAverage::from_period(3);

    // Feed sequence: [1, 2, 3]
    double seq[] = {1.0, 2.0, 3.0};
    double val = 0.0;
    for (double x: seq) {
        val = ema.update(x);
    }

    // Manual EMA calculation: with alpha=0.5
    // First value: 1.0 -> EMA1 = 1.0
    // EMA2 = 0.5*2 + 0.5*1 = 1.5
    // EMA3 = 0.5*3 + 0.5*1.5 = 2.25
    double expected = 2.25;
    double tol = 1e-12;

    if (std::fabs(val - expected) > tol) {
        std::cerr << "EMA_CHECK_FAILED: got " << val << " expected " << expected << "\n";
        return 1;
    }

    std::cout << "EMA_OK" << std::endl;
    return 0;
}
