#pragma once
#include <vector>
namespace pricing {
    double par_rate_example(const std::vector<double>& cashflows,
                            const std::vector<double>& discounts);
}