#include "time/calendar_factory.hpp"
#include "time/daycount.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

#include "instruments/swap.h"
#include "instruments/Leg.h"
#include "time/calendar_factory.hpp"
#include "time/daycount.hpp"
#include "curve/FlatRateCurve.h"
#include "instruments/FixFloatSwap.h"
#include "instruments/XCSwap.h"
#include "pricing/FixFloatSwapPricer.h"
//
// Created by Francisco Nunez on 23.11.2025.
//

using namespace curve::instruments;
using namespace curve::time;
using namespace std::chrono;

int main() {
    // Create supporting objects: calendar and daycount

    auto cal_ptr = curve::time::create_calendar(curve::time::FinancialCalendar::NYSE);
    auto dc = curve::time::create_daycount_convention(curve::time::DayCountConvention::ACT_360);

    Date cob = year{2025} / December / day{1};


    // Dates for the leg
    curve::time::Date start = cob + months{3};
    Date end = cob + months{18};

    // Construct two legs
    double notional = 1000000.0;
    std::chrono::months freq = std::chrono::months{6};

    Leg leg1(notional, "EUR", start, end, freq, *cal_ptr, BusinessDayConvention::FOLLOWING, *dc, Leg::FIXED);
    Leg leg2(notional, "EUR", start, end, freq, *cal_ptr, BusinessDayConvention::FOLLOWING, *dc, Leg::FLOATING);

    // Construct swap
    FixFloatSwap s(leg1, leg2);

    // name() should be currency1/currency2
    assert(s.name() == "swap");

    // Test par rate calculation
    std::shared_ptr<curve::ICurve> discount_curve = std::make_shared<curve::FlatRateCurve>(cob, 0.05);
    // 5% constant discount rate
    std::shared_ptr<curve::ICurve> forward_curve = std::make_shared<curve::FlatRateCurve>(cob, 0.05);
    // 5% constant forward rate


    // C++
    curve::market::MarketData *pmd = new curve::market::MarketData{
        .snap_time = std::chrono::sys_days(cob),
        .curves_ois = {
            {"EUR", discount_curve}, // use the correct shared_ptr element type here
        },
        .curves_funding = {{"EUR", forward_curve}}
    };

    std::shared_ptr<curve::market::MarketData> md(pmd);
    curve::pricing::FixFloatSwapPricer pricer;
    double rate = pricer.price(s, md);

    // With flat curves, the par rate should be very close to the forward rate.
    // Using an epsilon for floating point comparison.
    assert(std::abs(rate - 0.053) < 1e-2);
    std::cout << "PRICING_OK" << std::endl;
    return 0;
}
