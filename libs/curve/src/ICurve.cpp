#include "curve/ICurve.h"
//
// Created by Francisco Nunez on 14.11.2025.
//
#include "curve/ICurve.h"
#include "interpolation/linear.h"
#include "time/daycount.hpp"
using namespace curve;

ICurve::ICurve(std::vector<Pillar> &&pillars) : pillars_(std::move(pillars)) {
}

ICurve::ICurve(const std::vector<Pillar> &pillars) : pillars_(pillars) {
}

double ICurve::D(const std::chrono::days &t) const {
    if (pillars_.empty()) {
        throw std::runtime_error("No pillars to interpolate.");
    }

    if (t <= pillars_.front().get_dt()) return pillars_.front().get_value();
    if (t >= pillars_.back().get_dt()) return pillars_.back().get_value();

    auto u = std::upper_bound(pillars_.begin(), pillars_.end(), t, [](const std::chrono::days &x, const Pillar &p) {
        return x < p.get_dt();
    });

    auto d = std::lower_bound(pillars_.begin(), pillars_.end(), t, [](const Pillar &p, const std::chrono::days &x) {
        return p.get_dt() < x;
    });

    auto iu = u - pillars_.begin();
    auto id = d - pillars_.begin();

    double t1 = pillars_[id].get_dt().count();
    double t2 = pillars_[iu].get_dt().count();
    double v1 = pillars_[id].get_value();
    double v2 = pillars_[iu].get_value();

    auto dt = (t2 - t1);
    return v1 + (v2 - v1) * (t.count() - t1) / dt;
}

double ICurve::F(const time::Date t0, const std::chrono::days &dt1, const std::chrono::months &tenor,
                 const time::DayCountConventionBase &dc) const {
    auto t1 = std::chrono::sys_days(t0) + dt1;
    auto t2 = t1 + tenor;

    std::chrono::days tenorInDays = std::chrono::duration_cast<std::chrono::days>(t2 - t1);
    return F(t0, dt1, tenorInDays, dc);
}

double ICurve::F(const time::Date t0, const std::chrono::days &dt1, const std::chrono::days &tenor,
                 const time::DayCountConventionBase &dc) const {
    auto D1 = D(dt1);
    auto D2 = D(dt1 + tenor);
    if (D1 == D2) return 0.0;

    auto tanchor = std::chrono::sys_days(t0);
    auto tau = dc.year_fraction(tanchor + dt1, tanchor + dt1 + tenor);
    return (D1 / D2 - 1.0) / tau;
}

