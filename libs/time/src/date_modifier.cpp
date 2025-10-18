// filepath: libs/time/src/date_modifier.cpp
#include "time/date_modifier.hpp"
#include <stdexcept>

namespace curve::time {
    using namespace std::chrono;

    Date DateModifier::add_days(const Date &d, int days_count) {
        return Date{sys_days{d} + days{days_count}};
    }

    Date DateModifier::add_business_days(const Date &d, int business_days_count, const CalendarBase &cal) {
        Date date = d;
        if (business_days_count > 0) {
            for (int i = 0; i < business_days_count; ++i) {
                date = cal.next_business_day(date);
            }
        } else if (business_days_count < 0) {
            for (int i = 0; i < -business_days_count; ++i) {
                date = cal.prev_business_day(date);
            }
        }
        return date;
    }

    Date DateModifier::add_months(const Date &d, int months_count) {
        year_month_day ymd = d;
        months delta{months_count};

        // Compute target year_month
        year_month ym = year_month{ymd.year(), ymd.month()} + delta;

        // Try to keep the same day
        year_month_day candidate = year_month_day{ym / ymd.day()};
        if (candidate.ok()) return candidate;

        // Fallback to month-end: construct month_day_last from month
        month mo = ym.month();
        month_day_last mdl{mo};
        year_month_day_last ymdl = ym.year() / mdl;
        return year_month_day{sys_days{ymdl}};
    }

    Date DateModifier::add_years(const Date &d, int years_count) {
        year_month_day ymd = d;

        // Add years using chrono::years
        years delta_years{years_count};
        year y = ymd.year() + delta_years; // chrono::year + chrono::years
        year_month ym_target{y, ymd.month()};

        // Try to keep the same day
        year_month_day candidate = year_month_day{ym_target / ymd.day()};
        if (candidate.ok()) return candidate;

        // Fallback to month-end
        month mo = ym_target.month();
        month_day_last mdl{mo};
        year_month_day_last ymdl = ym_target.year() / mdl;
        return year_month_day{sys_days{ymdl}};
    }

    Date DateModifier::adjust(const Date &d, BusinessDayConvention conv, const CalendarBase &cal) {
        switch (conv) {
            case BusinessDayConvention::UNADJUSTED:
                return d;
            case BusinessDayConvention::FOLLOWING:
                if (!cal.is_holiday(d)) return d;
                return cal.next_business_day(d);
            case BusinessDayConvention::PRECEDING:
                if (!cal.is_holiday(d)) return d;
                return cal.prev_business_day(d);
            case BusinessDayConvention::MODIFIED_FOLLOWING: {
                if (!cal.is_holiday(d)) return d;
                Date f = cal.next_business_day(d);
                if (f.month() != d.month()) {
                    return cal.prev_business_day(d);
                }
                return f;
            }
            case BusinessDayConvention::MODIFIED_PRECEDING: {
                if (!cal.is_holiday(d)) return d;
                Date p = cal.prev_business_day(d);
                if (p.month() != d.month()) {
                    return cal.next_business_day(d);
                }
                return p;
            }
            default:
                throw std::invalid_argument("Unknown business day convention");
        }
    }
} // namespace curve::time
