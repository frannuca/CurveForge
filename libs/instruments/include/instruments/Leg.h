//
// Created by Francisco Nunez on 18.10.2025.
//

#ifndef CURVEFORGE_LEG_H
#define CURVEFORGE_LEG_H
#include "time/calendars.hpp"
#include "time/date.hpp"
#include "time/date_modifier.hpp"

namespace curve::instruments {
    class Leg {
    public:
        enum LegType {
            FIXED, FLOATING
        };

        Leg(const double &notional, std::string currency, time::Date start_date, time::Date end_date,
            const std::chrono::months &freq_monhts, const time::CalendarBase &calendar,
            const time::BusinessDayConvention &bdc, const LegType &leg_type);

        const std::vector<time::Date> &cashflows_dates() const;

    private:
        std::vector<time::Date> cashflows_;
        const time::CalendarBase &calendar;
        const int freq_payment_months;
        const time::Date start_date;
        const time::Date end_date;
        const double notional;
        const std::string currency;
        const time::BusinessDayConvention &bdc_;
        LegType leg_type_;
    };
}
#endif //CURVEFORGE_LEG_H
