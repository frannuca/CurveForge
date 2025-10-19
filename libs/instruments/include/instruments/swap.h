#pragma once

#include "instruments/Instrument.h"
#include <vector>

#include "time/calendars.hpp"
#include "time/date_modifier.hpp"

namespace curve::time {
    enum class FinancialCalendar;
}

namespace curve::instruments {
    class Swap : public Instrument {
    public:
        Swap() = delete;

        Swap(double notional, time::Date start_date, time::Date maturity, std::chrono::months leg1_freq,
             std::chrono::months leg2_freq, std::shared_ptr<time::CalendarBase> leg1_calendar,
             std::shared_ptr<time::CalendarBase> leg2_calendar,
             time::BusinessDayConvention bdc);

        ~Swap() override = default;

        std::string name() const override;

        const std::vector<time::Date> &get_leg1_payment_dates() const;

        const std::vector<time::Date> &get_leg2_payment_dates() const;

        double fixedRate() const;

    private:
        std::vector<time::Date> fixedSchedule_;
        std::vector<time::Date> floatSchedule_;
        std::chrono::months leg1_freq_, leg2_freq_;
        std::shared_ptr<time::CalendarBase> leg1_calendar_, &leg2_calendar_;
        double fixedRate_;
        std::vector<time::Date> leg1_payment_dates, leg2_payment_dates;
    };


    class SwapBuilder {
    public:
        SwapBuilder &withNotional(double notional) {
            notional_.emplace(notional);
            return *this;
        } ;

        SwapBuilder &withStartDate(const time::Date &start_date) {
            start_date_.emplace(start_date);
            return *this;
        };

        SwapBuilder &withMaturity(const time::Date &maturity) {
            maturity_.emplace(maturity);
            return *this;
        };

        SwapBuilder &withLeg1Frequency(const std::chrono::months &freq) {
            leg1_freq.emplace(freq);
            return *this;
        };

        SwapBuilder &withLeg2Frequency(const std::chrono::months &freq) {
            leg2_freq.emplace(freq);
            return *this;
        };

        SwapBuilder &withLeg1Calendar(const time::FinancialCalendar &calendar) {
            leg1_calendar.emplace(calendar);
            return *this;
        };

        SwapBuilder &withLeg2Calendar(const time::FinancialCalendar &calendar) {
            leg2_calendar.emplace(calendar);
            return *this;
        };;

        SwapBuilder &withBusinessDayConvention(time::BusinessDayConvention bdc) {
            bdc_.emplace(bdc);
            return *this;
        };

        std::shared_ptr<Swap> build();

    private:
        std::optional<double> notional_;
        std::optional<time::Date> start_date_, maturity_;
        std::optional<std::chrono::months> leg1_freq, leg2_freq;
        std::optional<time::FinancialCalendar> leg1_calendar, leg2_calendar;
        std::optional<time::BusinessDayConvention> bdc_ = time::BusinessDayConvention::UNADJUSTED;
    };
} // namespace instruments
