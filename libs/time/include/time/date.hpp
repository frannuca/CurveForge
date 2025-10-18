#ifndef TIME_DATE_HPP
#define TIME_DATE_HPP
#include <chrono>

namespace curve::time {
    using Date = std::chrono::year_month_day;

    inline std::string to_string(std::chrono::year_month_day ymd) {
        // %F -> YYYY-MM-DD; need a time_point (sys_days) for chrono formatting
        return std::format("{:%F}", std::chrono::sys_days{ymd});
    }
}

#endif // TIME_DATE_HPP
