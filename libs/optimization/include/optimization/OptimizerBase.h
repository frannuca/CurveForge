//
// Created by Francisco Nunez on 17.01.2026.
//

#ifndef CURVEFORGE_OPTIMIZERBASE_H
#define CURVEFORGE_OPTIMIZERBASE_H
#include "OptSolution.h"
#include <optional>
#include <functional>
#include "OptAlgoParams.h"
#include <cstdint>

namespace forge::optimization {
    enum class OptAlgorithm {
        AugLag,
        NelderMead,
        SimulatedAnnealing,
        ParticleSwarm,
        CMA_ES,
        GeneticAlgorithm
    };

    class OptimizerBase {
    public:
        OptimizerBase(std::function<double(const std::vector<double> &)> f,
                      std::optional<std::function<const std::vector<double>(const std::vector<double> &)> > df =
                              std::nullopt);

        virtual OptSolution solve(size_t n, const std::optional<std::vector<double> > &x0,
                                  std::vector<std::pair<double, double> > bounds = {},
                                  OptAlgoParams opt_algo_params = OptAlgoParams(1e-12, 1e-12, 200),
                                  std::optional<uint32_t> seed = std::nullopt) = 0;

    protected:
        const std::function<double(const std::vector<double> &)> objective_;
        std::optional<std::function<std::vector<double>(const std::vector<double> &)> > df_;
    };
}


#endif //CURVEFORGE_OPTIMIZERBASE_H
