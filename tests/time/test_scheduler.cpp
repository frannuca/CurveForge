//
// Created by Francisco Nunez on 18.10.2025.
//

#include "time/scheduler.h"
#include <chrono>
#include <iostream>

#include "time/calendar_factory.hpp"
using std::chrono::year;
using std::chrono::month;

using std::chrono::day;
/*
*Date start_date, Date end_date,
                                              const std::chrono::months &freq_monhts,
                                              const BusinessDayConvention &bdc, const DayCountConventionBase &dc,
                                              const CalendarBase &calendar)
 **/
int main() {
    auto schedule_vector = curve::time::Scheduler::generate_schedule(
        year{2025} / 10 / 1,
        year{2035} / 10 / 6,
        std::chrono::months{3},
        curve::time::BusinessDayConvention::FOLLOWING,
        *curve::time::create_daycount_convention(curve::time::DayCountConvention::ACT_360),
        *(curve::time::create_calendar(curve::time::FinancialCalendar::NYSE))
    );
    if (schedule_vector.accruals.size() != 14) {
        return 1;
    }
    std::cout << "CALENDARS_OK SCHEDULERtest";
}
