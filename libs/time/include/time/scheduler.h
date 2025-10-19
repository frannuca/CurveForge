//
// Created by Francisco Nunez on 18.10.2025.
//

#ifndef CURVEFORGE_SCHEDULER_H
#define CURVEFORGE_SCHEDULER_H
#include "calendars.hpp"
#include "date_modifier.hpp"
#include "time/date.hpp"

namespace curve {
    namespace time {
        class Scheduler {
        public:
            static std::vector<Date> generate_schedule(Date start_date, Date end_date,
                                                       const std::chrono::months &freq_monhts,
                                                       const CalendarBase &calendar, const BusinessDayConvention &bdc,
                                                       bool include_start_date);
        };
    } // time
} // curve

#endif //CURVEFORGE_SCHEDULER_H
