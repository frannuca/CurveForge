#include "instruments/swap.h"

#include "time/calendar_factory.hpp"
#include "time/scheduler.h"

namespace curve::instruments {
    Swap::Swap(double notional, time::Date start_date, time::Date maturity, std::chrono::months leg1_freq,
               std::chrono::months leg2_freq, std::shared_ptr<time::CalendarBase> leg1_calendar,
               std::shared_ptr<time::CalendarBase> leg2_calendar,
               time::BusinessDayConvention bdc) : Instrument(notional),
                                                  leg1_freq_(leg1_freq),
                                                  leg2_freq_(leg2_freq), leg1_calendar_(leg1_calendar),
                                                  leg2_calendar_(leg2_calendar) {
        leg1_payment_dates = time::Scheduler::generate_schedule(start_date, maturity, leg1_freq, *leg1_calendar_.get(),
                                                                bdc,
                                                                true);
        leg2_payment_dates = time::Scheduler::generate_schedule(start_date, maturity, leg2_freq, *leg1_calendar_.get(),
                                                                bdc,
                                                                true);
    }

    const std::vector<time::Date> &Swap::get_leg1_payment_dates() const { return leg1_payment_dates; }

    const std::vector<time::Date> &Swap::get_leg2_payment_dates() const { return leg2_payment_dates; }

    std::string Swap::name() const {
        return "Swap";
    }

    std::shared_ptr<Swap> SwapBuilder::build() {
        if (!notional_.has_value() || !start_date_.has_value() || !maturity_.has_value() ||
            !leg1_freq.has_value() || !leg2_freq.has_value() || !leg1_calendar.has_value() ||
            !leg2_calendar.has_value()) {
            throw std::runtime_error("Missing required fields to build Swap");
        }
        auto leg1_calendar_resolved = curve::time::create_calendar(leg1_calendar.value());
        auto leg2_calendar_resolved = curve::time::create_calendar(leg2_calendar.value());
        return std::make_shared<Swap>(notional_.value(), start_date_.value(), maturity_.value(), leg1_freq.value(),
                                      leg2_freq.value(), leg1_calendar_resolved, leg2_calendar_resolved, bdc_.value());
    }
} // namespace instruments

