//
// Created by Francisco Nunez on 04.10.2025.
//
#include <cmath>
#include "pricing/curve/curvebase.h"

using namespace pricing;

double curvebase::get_forward(double t, double dT) const {
    auto dfT = discount(t + dT);
    auto dfT_1 = discount(t);
    return dfT / dfT_1;
}

double curvebase::instantaneous_forward(double t) const {
    auto dft = get_forward(t, 1.0 / 250.0);
    auto dfT = discount(t);
    auto f = std::log(dft / dfT) / (1.0 / 250.0);
    return f;
}
