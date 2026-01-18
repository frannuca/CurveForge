//
// Created by Francisco Nunez on 18.01.2026.
//

#ifndef CURVEFORGE_OPTSOLUTION_H
#define CURVEFORGE_OPTSOLUTION_H
#include <string>
#include <utility>
#include <vector>
#include <ostream>

namespace forge::optimization {
    struct OptSolution {
        const double objective;
        const int id;
        const bool feasible;
        const std::vector<std::pair<std::string, double> > optimal_parameters;

        constexpr OptSolution(double objective_, int id_, bool feasible_,
                              std::vector<std::pair<std::string, double> > optimal_parameters_) noexcept
            : objective(objective_), id(id_), feasible(feasible_), optimal_parameters(std::move(optimal_parameters_)) {
        }

        constexpr OptSolution(const OptSolution &) = default;

        constexpr OptSolution(OptSolution &&) noexcept = default;

        OptSolution &operator=(const OptSolution &) = delete;

        OptSolution &operator=(OptSolution &&) = delete;

        ~OptSolution() = default;

        constexpr bool operator==(const OptSolution &other) const noexcept = default;
    };

    // Inline stream output for debugging/logging. Produces a compact representation:
    // OptSolution{id=..., objective=..., feasible=true/false, params=[(name,val), ...]}
    inline std::ostream &operator<<(std::ostream &os, const OptSolution &s) {
        os << "OptSolution{id=" << s.id
                << ", objective=" << s.objective
                << ", feasible=" << (s.feasible ? "true" : "false")
                << ", params=[";
        for (size_t i = 0; i < s.optimal_parameters.size(); ++i) {
            if (i) os << ", ";
            os << '(' << s.optimal_parameters[i].first << ", " << s.optimal_parameters[i].second << ')';
        }
        os << "]}";
        return os;
    }
}
#endif //CURVEFORGE_OPTSOLUTION_H
