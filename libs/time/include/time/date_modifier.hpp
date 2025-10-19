// ...existing code...
// filepath: libs/time/include/time/date_modifier.hpp
#pragma once

#include "date.hpp"
#include "calendars.hpp"
#include <chrono>

namespace curve::time {
    enum class BusinessDayConvention {
        UNADJUSTED,
        FOLLOWING,
        MODIFIED_FOLLOWING,
        PRECEDING,
        MODIFIED_PRECEDING
    };

    class DateModifier {
    public:
        using Date = curve::time::Date;

        // Add/subtract days (cheap)
        static Date add_days(const Date &d, int days_count);

        static Date add_business_days(const Date &d, int business_days_count, const CalendarBase &cal);

        // Add months/years, preserving day where possible; if resulting day is invalid, use month-end
        static Date add_months(const Date &d, std::chrono::months months_count);

        static Date add_years(const Date &d, int years_count);

        // Business day adjustments
        static Date adjust(const Date &d, BusinessDayConvention conv, const CalendarBase &cal);

        // Convenience helpers
        static Date following(const Date &d, const CalendarBase &cal) {
            return adjust(d, BusinessDayConvention::FOLLOWING, cal);
        }

        static Date modified_following(const Date &d, const CalendarBase &cal) {
            return adjust(d, BusinessDayConvention::MODIFIED_FOLLOWING, cal);
        }

        static Date preceding(const Date &d, const CalendarBase &cal) {
            return adjust(d, BusinessDayConvention::PRECEDING, cal);
        }

        static Date modified_preceding(const Date &d, const CalendarBase &cal) {
            return adjust(d, BusinessDayConvention::MODIFIED_PRECEDING, cal);
        }
    };
} // namespace curve::time
// ...existing code...
