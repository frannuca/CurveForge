//
// Created by Francisco Nunez on 18.10.2025.
//

#include "time/scheduler.h"
#include  "time/date_modifier.hpp"

namespace curve {
    namespace time {
        std::vector<Date> Scheduler::generate_schedule(Date start_date, Date end_date,
                                                       const int &freq_monhts,
                                                       CalendarBase &calendar, const BusinessDayConvention &bdc,
                                                       bool include_start_date) {
            if (start_date > end_date) { throw std::invalid_argument("start_date > end_date"); }

            std::vector<Date> schedule;
            auto modified_end_date = DateModifier::adjust(end_date, bdc, calendar);
            Date current_date = modified_end_date;
            schedule.push_back(modified_end_date);

            while (current_date > start_date) {
                current_date =
                        DateModifier::adjust(DateModifier::add_months(current_date, -freq_monhts), bdc, calendar);
                schedule.push_back(current_date);
            }

            if (include_start_date) schedule.push_back(start_date);
            std::sort(schedule.begin(), schedule.end(), [](auto a, auto b) { return a < b; });
            return schedule;
        }
    } // time
} // curve
