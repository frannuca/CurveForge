//
// Created by Francisco Nunez on 17.01.2026.
//

#include <optional>
#include <stdexcept>
#include <memory>
#include <limits>
#include <string>
#include <random>

#include "optimization/Convex_Boxed_Optimizer.h"
#include <nlopt.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace forge::optimization;

// Helper struct to pass C++ callables to NLopt via void* data
struct NLData {
    std::function<double(const std::vector<double> &)> objective;
    std::optional<std::function<const std::vector<double>(const std::vector<double> &)> > df;
};

namespace GradConstants {
    const double dx = 1e-9;
}

namespace {
    // Non-capturing wrapper compatible with NLopt function pointer type
    double nlopt_objective_wrapper(const std::vector<double> &x, std::vector<double> &grad, void *data) {
        NLData *d = static_cast<NLData *>(data);
        if (!d)
            throw std::runtime_error(
                "NLopt data pointer is null. It is needed to pass objective and gradient functioanl types as part of the data.");

        // If gradient requested and a gradient function was provided, fill it
        if (!grad.empty() && d->df.has_value()) {
            const std::vector<double> g = d->df.value()(x);
            const size_t nn = std::min(g.size(), grad.size());
            for (size_t i = 0; i < nn; ++i) grad[i] = g[i];
        } else if (!grad.empty() && !d->df.has_value()) {
            for (int i = 0; i < grad.size(); ++i) {
                std::vector<double> xbumped = x;
                xbumped[i] += GradConstants::dx;
                auto fp = d->objective(xbumped);
                xbumped[i] -= 2 * GradConstants::dx;
                auto fm = d->objective(xbumped);

                grad[i] = (fp - fm) / (2 * GradConstants::dx);
            }
        }

        // Evaluate objective
        return d->objective(x);
    }

    OptSolution internal_solve(nlopt::algorithm algo,
                               const std::function<double(const std::vector<double> &)> objective_,
                               std::optional<std::function<std::vector<double>(const std::vector<double> &)> > df_,
                               size_t n,
                               const std::optional<std::vector<double> > &x0,
                               std::vector<std::pair<double, double> > bounds,
                               OptAlgoParams opt_algo_params,
                               std::optional<uint32_t> seed) {
        try {
            // Outer optimizer: AUGLAG (handles constraints via augmented Lagrangian)
            nlopt::opt opt(algo, n);

            // Inner (local) optimizer: must be unconstrained or handle bounds well.
            // Use the Nocedal implementation of L-BFGS (LD_LBFGS_NOCEDAL).
            // Some nlopt builds disable the Luksan implementation (LD_LBFGS); attempting
            // to use it will fail with "Luksan code disabled". The Nocedal variant
            // is more widely available and doesn't require Luksan code.
            nlopt::opt local_opt(nlopt::LD_SLSQP, n);

            // Stopping criteria for inner solves (subproblems)
            local_opt.set_ftol_rel(opt_algo_params.ftol);
            local_opt.set_xtol_rel(opt_algo_params.xtol);
            local_opt.set_maxeval(opt_algo_params.maxeval);

            // Provide inner optimizer to AUGLAG
            opt.set_local_optimizer(local_opt);

            // Bounds: x >= 0, y >= 0
            std::vector<double> lb, ub;
            std::ranges::transform(bounds, std::back_inserter(lb), [](const auto &b) { return b.first; });
            std::ranges::transform(bounds, std::back_inserter(ub), [](const auto &b) { return b.second; });

            opt.set_lower_bounds(lb);
            opt.set_upper_bounds(ub);

            // Create random initial guess if none provided: sample each component uniformly in [lb[i], ub[i]].
            std::vector<double> x;
            if (x0.has_value()) {
                x = x0.value();
            } else {
                x.resize(n);
                // Deterministic seeding: use provided seed if present, otherwise use a fixed default seed
                constexpr uint32_t DEFAULT_SEED = 123456789u;
                uint32_t seed_val = seed.has_value() ? seed.value() : DEFAULT_SEED;
                std::mt19937 gen(seed_val);
                for (size_t i = 0; i < n; ++i) {
                    double l = (i < lb.size() ? lb[i] : 0.0);
                    double u = (i < ub.size() ? ub[i] : 1.0);
                    if (u < l) std::swap(l, u);
                    std::uniform_real_distribution<double> dist(l, u);
                    x[i] = dist(gen);
                }
            }

            // Prepare NLopt-compatible data and wrapper (cannot use capturing lambda)
            auto nl_data = std::make_unique<NLData>(NLData{objective_, df_});

            // Objective: use non-capturing wrapper and pass nl_data.get() as void*
            opt.set_min_objective(nlopt_objective_wrapper, nl_data.get());

            // Stopping criteria for outer AUGLAG iterations
            opt.set_ftol_rel(opt_algo_params.ftol);
            opt.set_xtol_rel(opt_algo_params.xtol);
            opt.set_maxeval(opt_algo_params.maxeval);

            double minf = 0.0;
            nlopt::result result = opt.optimize(x, minf);

            std::cout << "Result code: " << static_cast<int>(result) << "\n";
            std::cout << "x = " << x[0] << ", y = " << x[1] << "\n";
            std::cout << "f(x) = " << minf << "\n";

            // Build optimal parameters vector (name them x0, x1, ...)
            std::vector<std::pair<std::string, double> > optimal_params;
            for (size_t i = 0; i < x.size(); ++i) {
                optimal_params.emplace_back(std::string("x") + std::to_string(i), x[i]);
            }

            bool feasible = (static_cast<int>(result) > 0);
            return OptSolution(minf, static_cast<int>(result), feasible, std::move(optimal_params));
        } catch (const std::exception &e) {
            std::cerr << "NLopt error: " << e.what() << "\n";
            // Return a non-feasible solution with a sentinel id
            return OptSolution(std::numeric_limits<double>::quiet_NaN(), -1, false, {});
        }
    }
}

OptimizerBase::OptimizerBase(std::function<double(const std::vector<double> &)> f,
                             std::optional<std::function<const std::vector<double>(
                                 const std::vector<double> &)> > df) : objective_(f), df_(df) {
}

Convex_Boxed_Optimizer::Convex_Boxed_Optimizer(BoxedGradientBasedAlgos algo_,
                                               std::function<double(const std::vector<double> &)> f,
                                               std::optional<std::function<const std::vector<double>(
                                                   const std::vector<double> &)> > df) : algo(algo_),
    OptimizerBase(f, df) {
}

OptSolution Convex_Boxed_Optimizer::solve(size_t n, const std::optional<std::vector<double> > &x0,
                                          std::vector<std::pair<double, double> > bounds,
                                          OptAlgoParams opt_algo_params,
                                          std::optional<uint32_t> seed) {
    auto resolved_algo = nlopt::algorithm();
    switch (algo) {
        case BoxedGradientBasedAlgos::LD_AUGLAG:
            resolved_algo = nlopt::LD_AUGLAG;
            break;
        default:
            throw std::invalid_argument("Unsupported algorithm");
    }

    return internal_solve(resolved_algo, objective_, df_, n, x0, bounds, opt_algo_params, seed);
}
