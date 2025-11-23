//
// Created by Francisco Nunez on 22.11.2025.
//

#ifndef CURVEFORGE_GREEKS_H
#define CURVEFORGE_GREEKS_H
#include <optional>

#include <Eigen/Dense>

namespace curve::pricing {
    struct Greeks {
        std::optional<double> price;
        std::optional<double> pv;
        std::optional<double> delta;
        std::optional<double> gamma;
        std::optional<double> vega;
        std::optional<double> theta;
        std::optional<double> rho;
        std::optional<double> dv01;
        std::optional<Eigen::MatrixXd> cross_gamma;
    };
}
#endif //CURVEFORGE_GREEKS_H
