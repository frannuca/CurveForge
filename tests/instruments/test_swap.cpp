#include <cassert>
#include <iostream>
#include <cmath>

#include "instruments/swap.h"
#include "instruments/Leg.h"
#include "time/calendar_factory.hpp"
#include "time/daycount.hpp"
#include "mock_curve.h"

using namespace curve::instruments;
using namespace curve::time;
using namespace std::chrono;

int main() {
    // Create supporting objects: calendar and daycount
    auto cal_ptr = create_calendar(FinancialCalendar::NYSE);
    auto dc = create_daycount_convention(DayCountConvention::ACT_360);

    // Dates for the leg
    Date start = year{2025} / January / day{1};
    Date end = year{2027} / January / day{1};

    // Construct two legs
    double notional = 1000000.0;
    std::chrono::months freq = std::chrono::months{6};

    Leg leg1(notional, "USD", start, end, freq, *cal_ptr, BusinessDayConvention::FOLLOWING, *dc, Leg::FIXED);
    Leg leg2(notional * 2.2, "EUR", start, end, freq, *cal_ptr, BusinessDayConvention::FOLLOWING, *dc, Leg::FLOATING);

    // Construct swap
    Swap s(leg1, leg2);

    // name() should be currency1/currency2
    assert(s.name() == "swap");

    // Test par rate calculation
    MockCurve discount_curve(0.05); // 5% constant discount rate
    MockCurve forward_curve(0.05); // 5% constant forward rate

    double rate = s.par_rate(discount_curve, forward_curve);

    // With flat curves, the par rate should be very close to the forward rate.
    // Using an epsilon for floating point comparison.
    assert(std::abs(rate - 0.02) < 1e-2);

    std::cout << "SWAP_OK" << std::endl;
    return 0;
}
