//
// Created by Francisco Nunez on 18.10.2025.
//

#ifndef CURVEFORGE_SCHEDULER_H
#define CURVEFORGE_SCHEDULER_H
#include "calendars.hpp"
#include "date_modifier.hpp"
#include "time/date.hpp"
#include "time/daycount.hpp"

namespace curve {
    namespace time {
        struct AccruedPeriod {
            Date start_date;
            Date end_date;
            double accrual;
        };

        struct Schedule {
            Schedule(const std::vector<AccruedPeriod> &accruals, const std::chrono::months &freq_monhts,
                     const BusinessDayConvention &bdc, const DayCountConventionBase &dc,
                     const CalendarBase &calendar) : accruals(accruals), freq_monhts(freq_monhts), bdc(bdc), dc(dc),
                                                     calendar(calendar) {
            }

            const std::chrono::months &freq_monhts;
            const BusinessDayConvention &bdc;
            const DayCountConventionBase &dc;
            const CalendarBase &calendar;
            const std::vector<AccruedPeriod> accruals;
        };

        class Scheduler {
        public:
            static Schedule generate_schedule(Date start_date, Date end_date,
                                              const std::chrono::months &freq_monhts,
                                              const BusinessDayConvention &bdc, const DayCountConventionBase &dc,
                                              const CalendarBase &calendar);
        };
    } // time
} // curve

#endif //CURVEFORGE_SCHEDULER_H
