#include "instruments/swap.h"

#include "instruments/Leg.h"
#include "time/calendar_factory.hpp"
#include "time/scheduler.h"

namespace curve::instruments {
    const time::Schedule &Swap::get_leg1_payment_dates() const { return leg1_.cashflows_schedule(); }

    const time::Schedule &Swap::get_leg2_payment_dates() const { return leg2_.cashflows_schedule(); }


    std::string Swap::name() const {
        return "swap";
    }
} // namespace instruments

