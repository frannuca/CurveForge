//
// Created by Francisco Nunez on 18.10.2025.
//

#include "time/scheduler.h"
#include  "time/date_modifier.hpp"

namespace curve {
    namespace time {
        Schedule Scheduler::generate_schedule(Date start_date, Date end_date,
                                              const std::chrono::months &freq_monhts,
                                              const BusinessDayConvention &bdc, const DayCountConventionBase &dc,
                                              const CalendarBase &calendar) {
            if (start_date > end_date) { throw std::invalid_argument("start_date > end_date"); }

            std::vector<AccruedPeriod> accruals;

            auto modified_end_date = DateModifier::adjust(end_date, bdc, calendar);
            Date current_date = modified_end_date;
            while (current_date > start_date) {
                current_date =
                        DateModifier::adjust(DateModifier::add_months(current_date, -freq_monhts), bdc, calendar);
                double accrued = dc.year_fraction(current_date, modified_end_date);
                accruals.push_back({current_date, modified_end_date, accrued});
                modified_end_date = current_date;
            }
            std::sort(accruals.begin(), accruals.end(), [](const AccruedPeriod &a, const AccruedPeriod &b) {
                return a.start_date < b.start_date;
            });
            return {accruals, freq_monhts, bdc, dc, calendar};
        }
    } // time
} // curve
