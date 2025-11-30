#include <cassert>
#include <iostream>
#include "time/date_modifier.hpp"
#include "time/calendars.hpp"
#include "time/calendar_factory.hpp"
#include <chrono>

using namespace curve::time;
using namespace std::chrono;

int main() {
    // Basic add_days
    Date d = year{2025} / October / day{18};
    Date plus10 = DateModifier::add_days(d, 10);
    assert(sys_days{plus10} - sys_days{d} == days{10});

    // add_months edge-case: Jan 31 -> Feb 28 (2025 not a leap year)
    Date jan31 = year{2025} / January / day{31};
    Date febTarget = DateModifier::add_months(jan31, std::chrono::months{1});
    Date expectedFeb = year{2025} / February / day{28};
    assert(sys_days{febTarget} == sys_days{expectedFeb});

    // Business day following with NYSE calendar (July 4 is holiday)
    auto cal = create_calendar(FinancialCalendar::NYSE);
    Date independenceDay = year{2023} / July / day{4};
    assert(cal->is_holiday(independenceDay));
    Date foll = DateModifier::following(independenceDay, *cal);
    // foll must be a business day (not holiday)
    assert(!cal->is_holiday(foll));
    // and be after the holiday
    assert(sys_days{foll} > sys_days{independenceDay});

    std::cout << "CALENDAR_OK" << std::endl;
    return 0;
}
