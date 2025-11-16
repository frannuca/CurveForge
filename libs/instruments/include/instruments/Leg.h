//
// Created by Francisco Nunez on 18.10.2025.
//

#ifndef CURVEFORGE_LEG_H
#define CURVEFORGE_LEG_H
#include "Instrument.h"
#include "time/calendars.hpp"
#include "time/date.hpp"
#include "time/date_modifier.hpp"
#include "time/scheduler.h"

namespace curve::instruments {
    class Leg : public Instrument {
    public:
        enum LegType {
            FIXED, FLOATING
        };

        Leg(const double &notional, const std::string &currency, time::Date start_date, time::Date end_date,
            const std::chrono::months &payment_intervals, const time::CalendarBase &calendar,
            const time::BusinessDayConvention &bdc, const time::DayCountConventionBase &dc, const LegType &leg_type);

        [[nodiscard]] const time::Schedule &cashflows_schedule() const;

        [[nodiscard]] double notional() const;

        std::string name() const override;

    private:
        double notional_;
        time::Schedule schedule_;
        LegType leg_type_;
    };
}
#endif //CURVEFORGE_LEG_H
