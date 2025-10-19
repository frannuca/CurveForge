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

int main() {
    auto schedule_vector = curve::time::Scheduler::generate_schedule(
        year{2025} / 10 / 1,
        year{2035} / 10 / 6,
        std::chrono::months{3},
        *(curve::time::create_calendar(curve::time::FinancialCalendar::NYSE)),
        curve::time::BusinessDayConvention::FOLLOWING,
        true
    );
    if (schedule_vector.size() != 14) {
        return 1;
    }
    std::cout << "CALENDARS_OK SCHEDULERtest";
}
