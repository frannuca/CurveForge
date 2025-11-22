//
// Created by Francisco Nunez on 18.10.2025.
//
#include "instruments/Leg.h"
#include <future>
#include "time/scheduler.h"

using namespace curve::instruments;
using namespace curve::time;

Leg::Leg(const double &notional, const std::string &currency, Date start_date, time::Date end_date,
         const std::chrono::months &payment_intervals, const time::CalendarBase &calendar,
         const time::BusinessDayConvention &bdc, const time::DayCountConventionBase &dc,
         const LegType &leg_type) : Instrument(currency), notional_(notional), leg_type_(leg_type),
                                    schedule_(std::move(
                                        Scheduler::generate_schedule(start_date, end_date, payment_intervals, bdc, dc,
                                                                     calendar))) {
}

const Schedule &Leg::cashflows_schedule() const {
    return schedule_;
}

double Leg::notional() const {
    return notional_;
}

std::string Leg::name() const {
    return "Leg";
}

Leg::LegType Leg::leg_type() const {
    return leg_type_;
}
