//
// Created by Francisco Nunez on 18.01.2026.
//

#include <iostream>

#include "optimization/Convex_Boxed_Optimizer.h"
using namespace forge::optimization;

int main() {
    std::function<double(const std::vector<double> &)> fobj = [](std::vector<double> x) {
        return std::pow(x[0] - 1.33, 2) + std::pow(x[1] - 1.33, 2);
    };
    Convex_Boxed_Optimizer opt(BoxedGradientBasedAlgos::LD_AUGLAG, fobj, std::nullopt);

    auto s = opt.solve(2, std::nullopt, {{-10, 10}, {-10, 10}}, OptAlgoParams{1e-6, 1e-6, 50});
    std::cout << s << std::endl;
    std::cout << "OPT_OK" << std::endl;
    return 0;
}
