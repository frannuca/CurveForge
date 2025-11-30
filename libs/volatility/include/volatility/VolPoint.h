//
// Created by Francisco Nunez on 30.11.2025.
//

#ifndef CURVEFORGE_VOLPOINT_H
#define CURVEFORGE_VOLPOINT_H

namespace curve::volatility {
    /**
        * @brief Structure to hold calibrated volatility point
        */
    struct VolPoint {
        double strike;
        double maturity;
        double volatility;
        double moneyness;

        VolPoint(double s, double t, double v, double m = 0)
            : strike(s), maturity(t), volatility(v), moneyness(m) {
        }
    };
}
#endif //CURVEFORGE_VOLPOINT_H
