#include "curve/ICurve.h"
//
// Created by Francisco Nunez on 14.11.2025.
//
#include <utility>

#include "curve/ICurve.h"
#include "interpolation/linear.h"
#include "time/daycount.hpp"
using namespace curve;

ICurve::ICurve(const time::Date &cob_date,
               std::vector<Pillar> &&pillars,
               std::shared_ptr<time::DayCountConventionBase> convention) : pillars_(std::move(pillars)),
                                                                           cob_date(cob_date),
                                                                           dc(std::move(convention)) {
}

ICurve::ICurve(const time::Date &cob_date, const std::vector<Pillar> &pillars,
               std::shared_ptr<time::DayCountConventionBase> convention) : cob_date(cob_date), pillars_(pillars),
                                                                           dc(std::move(convention)) {
}

double ICurve::D(const time::Date &t_in) const {
    if (pillars_.empty()) {
        throw std::runtime_error("No pillars to interpolate.");
    }
    auto t = t_in;
    if (t <= pillars_.front().get_time()) t = pillars_.front().get_time();
    if (t >= pillars_.back().get_time()) t = pillars_.back().get_time();

    auto i_upper = std::upper_bound(pillars_.begin(), pillars_.end(), t, [](const time::Date &x, const Pillar &p) {
        return x < p.get_time();
    });

    auto i_lower = std::lower_bound(pillars_.begin(), pillars_.end(), t, [](const Pillar &p, const time::Date &x) {
        return p.get_time() > x;
    });

    auto iu = i_upper - pillars_.begin();
    auto id = i_lower - pillars_.begin();

    const auto t1 = pillars_[id].get_time();
    const auto t2 = pillars_[iu].get_time();

    const double v1 = pillars_[id].get_value();
    const double v2 = pillars_[iu].get_value();

    const double dt = dc->year_fraction(t1, t2);
    const double dT = dc->year_fraction(t1, t);

    const auto rate = v1 + (v2 - v1) * dT / dt;
    const auto t_cob = dc->year_fraction(cob_date, t);
    return std::exp(-rate * t_cob);
}

double ICurve::F(const time::Date &t1, const time::Date &t2) const {
    const auto D1 = D(t1);
    const auto D2 = D(t2);
    const auto tau = dc->year_fraction(t1, t2);

    return (D1 / D2 - 1.0) / tau;
}
