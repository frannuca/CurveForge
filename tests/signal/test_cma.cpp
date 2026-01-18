#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <iomanip>
#include "signal/CrossMovingAverage.h"

int main() {
    using namespace forge::signal;

    // short period reacts faster than long period
    CrossMovingAverage cma(2, 5);

    std::vector<double> seq;
    // Start with low values
    for (int i = 0; i < 5; ++i) seq.push_back(0.0);
    // Spike up to cause bullish crossover
    for (int i = 0; i < 6; ++i) seq.push_back(10.0);
    // Then drop to cause bearish crossover
    for (int i = 0; i < 10; ++i) seq.push_back(0.0);

    bool saw_bull = false;
    bool saw_bear = false;
    std::optional<double> prev_diff;

    // containers to record for plotting
    std::vector<double> out_sample;
    std::vector<double> out_short;
    std::vector<double> out_long;
    std::vector<double> out_diff;

    for (double x: seq) {
        double diff = cma.update(x); // short - long
        // record values: short and long may be available via accessors
        auto s_opt = cma.short_value();
        auto l_opt = cma.long_value();
        double s = s_opt.has_value() ? *s_opt : 0.0;
        double l = l_opt.has_value() ? *l_opt : 0.0;
        out_sample.push_back(x);
        out_short.push_back(s);
        out_long.push_back(l);
        out_diff.push_back(diff);

        if (prev_diff.has_value()) {
            double prev = *prev_diff;
            if (prev <= 0.0 && diff > 0.0) saw_bull = true;
            if (prev >= 0.0 && diff < 0.0) saw_bear = true;
        }
        prev_diff = diff;
    }

    // write CSV for plotting in the current working directory (when run by ctest this is the build dir)
    std::ofstream csv("signal_cma.csv");
    if (csv) {
        csv << "sample,short,long,diff\n";
        csv << std::fixed << std::setprecision(12);
        for (size_t i = 0; i < out_sample.size(); ++i) {
            csv << out_sample[i] << ',' << out_short[i] << ',' << out_long[i] << ',' << out_diff[i] << '\n';
        }
        csv.close();
        std::cout << "WROTE signal_cma.csv\n";
    } else {
        std::cerr << "WARNING: could not write signal_cma.csv\n";
    }

    if (!saw_bull) {
        std::cerr << "CMA_CHECK_FAILED: no bullish signal observed\n";
        return 1;
    }
    if (!saw_bear) {
        std::cerr << "CMA_CHECK_FAILED: no bearish signal observed\n";
        return 1;
    }

    std::cout << "CMA_OK" << std::endl;
    return 0;
}
