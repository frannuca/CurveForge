//
// Created by Francisco Nunez on 22.11.2025.
//

#include "instruments/XCSwap.h"

curve::instruments::XCSwap::XCSwap(const Leg &base_ccy_leg, const Leg &foreign_ccy_leg) : fxSpot_(
        base_ccy_leg.notional() / foreign_ccy_leg.notional()), Swap(
        base_ccy_leg, foreign_ccy_leg) {
    if (base_ccy_leg.leg_type() != Leg::FLOATING) {
        throw std::invalid_argument("base_ccy_leg leg must be floating.");
    }
    if (foreign_ccy_leg.leg_type() != Leg::FLOATING) {
        throw std::invalid_argument("foreign_ccy_leg leg must be floating.");
    }
}

double curve::instruments::XCSwap::fxSpot() const {
    return fxSpot_;
}
