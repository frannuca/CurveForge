//
// Created by Francisco Nunez on 18.01.2026.
//

#ifndef CURVEFORGE_CONVEX_AUGLAGOPTIMIZER_H
#define CURVEFORGE_CONVEX_AUGLAGOPTIMIZER_H
#include "optimization/OptimizerBase.h"

namespace forge::optimization {
    enum class BoxedGradientBasedAlgos : int {
        LD_LBFGS_B,
        LD_AUGLAG
    };

    class Convex_Boxed_Optimizer : public OptimizerBase {
    public:
        Convex_Boxed_Optimizer(BoxedGradientBasedAlgos algo_, std::function<double(const std::vector<double> &)> f,
                               std::optional<std::function<const std::vector<double>(
                                   const std::vector<double> &)> > df);


        OptSolution solve(size_t n, const std::optional<std::vector<double> > &x0,
                          std::vector<std::pair<double, double> > bounds,
                          OptAlgoParams opt_algo_params,
                          std::optional<uint32_t> seed = std::nullopt) override;

        const BoxedGradientBasedAlgos algo;
    };
}
#endif //CURVEFORGE_CONVEX_AUGLAGOPTIMIZER_H
