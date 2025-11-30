//
// Created by Francisco Nunez on 28.09.2025.
//
#include <cassert>
#include <iostream>
#include "time/calendars.hpp"
#include "time/calendar_factory.hpp"
#include <chrono>
using namespace curve::time;
using namespace std::chrono;
using namespace std::chrono_literals;

int main() {
    auto calendar = create_calendar(FinancialCalendar::NYSE);

    Date independenceDay = 2023y / 7 / 4;
    assert(calendar->is_holiday(independenceDay) && "July 4, 2023 should be a holiday (Independence Day)");

    std::cout << "CALENDAR_OK" << std::endl;
}
