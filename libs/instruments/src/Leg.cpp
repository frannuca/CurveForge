//
// Created by Francisco Nunez on 18.10.2025.
//
#include "instruments/Leg.h"
#include <future>
#include "time/scheduler.h"

using namespace curve::instruments;
using namespace curve::time;

Leg::Leg(const double &notional, std::string currency, Date start_date, time::Date end_date,
         const std::chrono::months &freq_monhts, const time::CalendarBase &calendar,
         const time::BusinessDayConvention &bdc, const LegType &leg_type) : calendar(calendar), freq_payment_months(0),
                                                                            start_date(start_date),
                                                                            end_date(end_date), notional(notional),
                                                                            bdc_(bdc), leg_type_(leg_type) {
    cashflows_ = std::move(Scheduler::generate_schedule(start_date, end_date, freq_monhts, calendar, bdc, true));
}

const std::vector<Date> &Leg::cashflows_dates() const {
    return cashflows_;
}
