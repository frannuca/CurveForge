//
// Created by Francisco Nunez on 23.11.2025.
//

#include "../include/pricing/GreekCalculator.h"
using namespace curve::pricing;

double GreekCalculator::dv01(const curve::instruments::Instrument &p) {
    throw std::runtime_error("Not implemented.");
}

double GreekCalculator::delta(const curve::instruments::Instrument &p) {
    throw std::runtime_error("Not implemented.");
}

double GreekCalculator::gamma(const curve::instruments::Instrument &p) {
    throw std::runtime_error("Not implemented.");
}
