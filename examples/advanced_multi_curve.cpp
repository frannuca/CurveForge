// advanced_multi_curve.cpp
// Single-file C++17 advanced scaffold with analytic Jacobians (direct), smoothness penalty,
// forward-curve bucket ladders, and simple CSV hooks (placeholders).
// No third-party deps.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

using namespace std;

/************* Day count & simple schedules *************/
enum class DayCount { ACT360, ACT365F, THIRTY_360 };

static double yf(double t0_days, double t1_days, DayCount dc) {
    double dt = max(0.0, t1_days - t0_days);
    switch (dc) {
        case DayCount::ACT360: return dt / 360.0;
        case DayCount::ACT365F: return dt / 365.0;
        case DayCount::THIRTY_360: return dt / 360.0;
    }
    return dt / 365.0;
}

struct Schedule {
    vector<double> times;
    vector<double> accruals;

    static Schedule regular(double maturity, int paymentsPerYear, DayCount dc) {
        Schedule s;
        int n = max(1, (int) ::round(maturity * paymentsPerYear));
        double dt = n == 1 ? maturity : 1.0 / paymentsPerYear;
        double prev = 0.0;
        for (int i = 1; i <= n; ++i) {
            double t = i * dt;
            s.times.push_back(t);
            s.accruals.push_back(yf(prev * 365.0, t * 365.0, dc));
            prev = t;
        }
        return s;
    }
};

/************* Linear interpolation & basis weights *************/
static double lininterp(const vector<double> &x, const vector<double> &y, double xq) {
    if (xq <= x.front()) return y.front();
    if (xq >= x.back()) return y.back();
    auto it = upper_bound(x.begin(), x.end(), xq);
    size_t j = size_t(it - x.begin()), i = j - 1;
    double w = (xq - x[i]) / (x[j] - x[i]);
    return y[i] * (1.0 - w) + y[j] * w;
}

static unordered_map<int, double> basis_weights_at(const vector<double> &x, double xq) {
    unordered_map<int, double> w;
    if (xq <= x.front()) {
        w[0] = 1.0;
        return w;
    }
    if (xq >= x.back()) {
        w[(int) x.size() - 1] = 1.0;
        return w;
    }
    auto it = upper_bound(x.begin(), x.end(), xq);
    int j = int(it - x.begin()), i = j - 1;
    double alpha = (xq - x[i]) / (x[j] - x[i]);
    w[i] = 1.0 - alpha;
    w[j] = alpha;
    return w;
}

/************* Curves *************/
struct DiscountCurve {
    virtual ~DiscountCurve() = default;

    virtual double D(double t) const = 0;
};

struct ForwardingCurve {
    virtual ~ForwardingCurve() = default;

    virtual double P(double t) const = 0;
};

struct PWLogDF : DiscountCurve {
    vector<double> knots, zeros, logDF, rates;

    PWLogDF() = default;

    PWLogDF(vector<double> k, vector<double> ratesi) : rates(std::move(ratesi)), knots(std::move(k)) {
        if (knots.empty() || fabs(knots[0]) > 1e-14) throw runtime_error("PWLogDF: first knot must be 0");
        if (rates.size() != knots.size()) throw runtime_error("PWLogDF: rates size must match knots");
        zeros.resize(rates.size());
        for (size_t i = 0; i < rates.size(); ++i) zeros[i] = rates[i] * knots[i];
        logDF.resize(knots.size());
        for (size_t i = 0; i < knots.size(); ++i) logDF[i] = -zeros[i];
    }

    double D(double t) const override {
        double z = lininterp(knots, zeros, t);
        return exp(-z);
    }

    // D(t) = exp(-z(t)), z = lininterp(zeros)
    // dD/dtheta_j = -D(t) * w_j
    unordered_map<int, double> dD_dtheta(double t) const {
        double z = lininterp(knots, zeros, t);
        double Dv = exp(-z);
        auto w = basis_weights_at(knots, t);
        for (auto &kv: w) kv.second *= -Dv;
        return w;
    }

    void set_last(double v) {
        rates.back() = v;
        zeros.back() = v * knots.back();
        logDF.back() = -v * knots.back();
    }
};

struct PWLogForwardDF : ForwardingCurve {
    vector<double> knots, zeros, logDF, rates;

    PWLogForwardDF() = default;

    PWLogForwardDF(vector<double> k, vector<double> ratesi) : rates(std::move(ratesi)), knots(std::move(k)) {
        if (knots.empty() || fabs(knots[0]) > 1e-14) throw runtime_error("PWLogForwardDF: first knot must be 0");
        if (rates.size() != knots.size()) throw runtime_error("PWLogForwardDF: rates size must match knots");
        zeros.resize(rates.size());
        for (size_t i = 0; i < rates.size(); ++i) zeros[i] = rates[i] * knots[i];
        logDF.resize(knots.size());
        for (size_t i = 0; i < knots.size(); ++i) logDF[i] = -zeros[i];
    }

    double P(double t) const override {
        double z = lininterp(knots, zeros, t);
        return exp(-z);
    }

    // P(t) = exp(-z_f(t)), z_f = lininterp(zeros)
    // dP/dphi_j = -P(t) * w_j
    unordered_map<int, double> dP_dphi(double t) const {
        double y = lininterp(knots, zeros, t);
        double Pval = exp(-y);
        auto w = basis_weights_at(knots, t);
        for (auto &kv: w) kv.second *= -Pval;
        return w;
    }

    void set_last(double v) {
        rates.back() = v;
        zeros.back() = v * knots.back();
        logDF.back() = -v * knots.back();
    }
};

/************* Forwards *************/
static double forward_simple(const PWLogForwardDF &F, double T0, double T1, double a) {
    double P0 = F.P(T0), P1 = F.P(T1);
    return (P0 / P1 - 1.0) / (a == 0.0 ? 1e-18 : a);
}

static unordered_map<int, double> d_forward_dphi(const PWLogForwardDF &F, double T0, double T1, double a) {
    unordered_map<int, double> out;
    double acc = (a == 0.0 ? 1e-18 : a);
    double P0 = F.P(T0), P1 = F.P(T1);
    auto dP0 = F.dP_dphi(T0);
    auto dP1 = F.dP_dphi(T1);
    for (auto &kv: dP0) out[kv.first] += (kv.second / P1) / acc;
    for (auto &kv: dP1) out[kv.first] += -(P0 / (P1 * P1)) * kv.second / acc;
    return out;
}

/************* Instruments *************/
struct OISSwap {
    Schedule fixed;
    double maturity;
    double quoteRate;
    DayCount dcFixed;

    static double par_rate(const PWLogDF &D, const Schedule &fixed) {
        double pv01 = 0.0;
        for (size_t j = 0; j < fixed.times.size(); ++j) pv01 += fixed.accruals[j] * D.D(fixed.times[j]);
        double TN = fixed.times.back();
        return (1.0 - D.D(TN)) / pv01;
    }

    double par_rate(const PWLogDF &D) const { return par_rate(D, fixed); }

    unordered_map<int, double> d_par_rate_dtheta(const PWLogDF &D) const {
        double pv01 = 0.0;
        for (size_t j = 0; j < fixed.times.size(); ++j) pv01 += fixed.accruals[j] * D.D(fixed.times[j]);
        double TN = fixed.times.back();
        double DTN = D.D(TN);
        unordered_map<int, double> d_pv01;
        for (size_t j = 0; j < fixed.times.size(); ++j) {
            auto dD = D.dD_dtheta(fixed.times[j]);
            for (auto &kv: dD) d_pv01[kv.first] += fixed.accruals[j] * kv.second;
        }
        auto dDTN = D.dD_dtheta(TN);
        unordered_map<int, double> out;
        unordered_set<int> idx;
        for (auto &kv: d_pv01) idx.insert(kv.first);
        for (auto &kv: dDTN) idx.insert(kv.first);
        for (int j: idx) {
            double dpv01 = d_pv01.count(j) ? d_pv01[j] : 0.0;
            double dnum = -(dDTN.count(j) ? dDTN[j] : 0.0);
            out[j] = (dnum * pv01 - (1.0 - DTN) * dpv01) / (pv01 * pv01);
        }
        return out;
    }
};

struct IRSSwap {
    Schedule fixed, floatSched;
    double quoteRate;
    DayCount dcFixed;

    double par_rate(const PWLogDF &D, const PWLogForwardDF &F) const {
        double pv01 = 0.0;
        for (size_t j = 0; j < fixed.times.size(); ++j) pv01 += fixed.accruals[j] * D.D(fixed.times[j]);
        double pvF = 0.0;
        double Tprev = 0.0;
        for (size_t i = 0; i < floatSched.times.size(); ++i) {
            double t = floatSched.times[i], a = floatSched.accruals[i];
            double L = forward_simple(F, Tprev, t, a);
            pvF += a * D.D(t) * L;
            Tprev = t;
        }
        return pvF / pv01;
    }

    unordered_map<int, double> d_par_rate_dtheta(const PWLogDF &D, const PWLogForwardDF &F) const {
        double pv01 = 0.0;
        for (size_t j = 0; j < fixed.times.size(); ++j) pv01 += fixed.accruals[j] * D.D(fixed.times[j]);
        double pvF = 0.0;
        double Tprev = 0.0;
        for (size_t i = 0; i < floatSched.times.size(); ++i) {
            double t = floatSched.times[i], a = floatSched.accruals[i];
            double L = forward_simple(F, Tprev, t, a);
            pvF += a * D.D(t) * L;
            Tprev = t;
        }
        unordered_map<int, double> d_pv01, d_pvF;
        for (size_t j = 0; j < fixed.times.size(); ++j) {
            auto dD = D.dD_dtheta(fixed.times[j]);
            for (auto &kv: dD) d_pv01[kv.first] += fixed.accruals[j] * kv.second;
        }
        Tprev = 0.0;
        for (size_t i = 0; i < floatSched.times.size(); ++i) {
            double t = floatSched.times[i], a = floatSched.accruals[i];
            auto dD = D.dD_dtheta(t);
            double L = forward_simple(F, Tprev, t, a);
            for (auto &kv: dD) d_pvF[kv.first] += a * L * kv.second;
            Tprev = t;
        }
        unordered_set<int> idx;
        for (auto &kv: d_pv01) idx.insert(kv.first);
        for (auto &kv: d_pvF) idx.insert(kv.first);
        unordered_map<int, double> out;
        for (int j: idx) {
            out[j] = ((d_pvF.count(j) ? d_pvF[j] : 0.0) * pv01 - pvF * (d_pv01.count(j) ? d_pv01[j] : 0.0)) / (
                         pv01 * pv01);
        }
        return out;
    }

    unordered_map<int, double> d_par_rate_dphi(const PWLogDF &D, const PWLogForwardDF &F) const {
        double pv01 = 0.0;
        for (size_t j = 0; j < fixed.times.size(); ++j) pv01 += fixed.accruals[j] * D.D(fixed.times[j]);
        unordered_map<int, double> d_pvF;
        double Tprev = 0.0;
        for (size_t i = 0; i < floatSched.times.size(); ++i) {
            double t = floatSched.times[i], a = floatSched.accruals[i];
            auto dL = d_forward_dphi(F, Tprev, t, a);
            for (auto &kv: dL) d_pvF[kv.first] += a * D.D(t) * kv.second;
            Tprev = t;
        }
        unordered_map<int, double> out;
        for (auto &kv: d_pvF) out[kv.first] = kv.second / pv01;
        return out;
    }
};

struct BasisSwap {
    Schedule leg1, leg2;
    double quoteSpread;

    double pv_diff(const PWLogDF &D, const PWLogForwardDF &F1, const PWLogForwardDF &F2) const {
        double pv1 = 0.0;
        double Tprev = 0.0;
        for (size_t i = 0; i < leg1.times.size(); ++i) {
            double t = leg1.times[i], a = leg1.accruals[i];
            double L = forward_simple(F1, Tprev, t, a);
            pv1 += a * D.D(t) * (L + quoteSpread);
            Tprev = t;
        }
        double pv2 = 0.0;
        Tprev = 0.0;
        for (size_t i = 0; i < leg2.times.size(); ++i) {
            double t = leg2.times[i], a = leg2.accruals[i];
            double L = forward_simple(F2, Tprev, t, a);
            pv2 += a * D.D(t) * L;
            Tprev = t;
        }
        return pv1 - pv2;
    }

    unordered_map<int, double> d_pv_diff_dtheta(const PWLogDF &D, const PWLogForwardDF &F1,
                                                const PWLogForwardDF &F2) const {
        unordered_map<int, double> out;
        double Tprev = 0.0;
        for (size_t i = 0; i < leg1.times.size(); ++i) {
            double t = leg1.times[i], a = leg1.accruals[i];
            double L = forward_simple(F1, Tprev, t, a);
            auto dD = D.dD_dtheta(t);
            for (auto &kv: dD) out[kv.first] += a * (L + quoteSpread) * kv.second;
            Tprev = t;
        }
        Tprev = 0.0;
        for (size_t i = 0; i < leg2.times.size(); ++i) {
            double t = leg2.times[i], a = leg2.accruals[i];
            double L = forward_simple(F2, Tprev, t, a);
            auto dD = D.dD_dtheta(t);
            for (auto &kv: dD) out[kv.first] -= a * L * kv.second;
            Tprev = t;
        }
        return out;
    }

    unordered_map<int, double> d_pv_diff_dphi(const PWLogDF &D, const PWLogForwardDF &F1, const PWLogForwardDF &F2,
                                              int which) const {
        unordered_map<int, double> out;
        if (which == 1) {
            double Tprev = 0.0;
            for (size_t i = 0; i < leg1.times.size(); ++i) {
                double t = leg1.times[i], a = leg1.accruals[i];
                auto dL = d_forward_dphi(F1, Tprev, t, a);
                for (auto &kv: dL) out[kv.first] += a * D.D(t) * kv.second;
                Tprev = t;
            }
        } else {
            double Tprev = 0.0;
            for (size_t i = 0; i < leg2.times.size(); ++i) {
                double t = leg2.times[i], a = leg2.accruals[i];
                auto dL = d_forward_dphi(F2, Tprev, t, a);
                for (auto &kv: dL) out[kv.first] -= a * D.D(t) * kv.second;
                Tprev = t;
            }
        }
        return out;
    }
};

/************* Root finder & bootstrap *************/
struct Root {
    template<class F>
    static double bracketed_secant(F func, double a, double b, double target,
                                   int iters = 100, double tol = 1e-12) {
        auto g = [&](double x) { return func(x) - target; };
        double fa = g(a), fb = g(b);
        if (!(fa * fb <= 0.0)) {
            // try to expand once if not bracketed
            double da = a - (b - a), db = b + (b - a);
            double fda = g(da), fdb = g(db);
            if (fda * fb <= 0) {
                a = da;
                fa = fda;
            } else if (fa * fdb <= 0) {
                b = db;
                fb = fdb;
            } else throw runtime_error("Root: interval not bracketed");
        }
        double xL = a, xR = b, fL = fa, fR = fb;
        double x = 0.5 * (xL + xR), f = g(x);
        for (int i = 0; i < iters; ++i) {
            if (std::abs(f) < tol || std::abs(xR - xL) < tol) return x;
            // regula falsi step
            double xRF = xR - fR * (xR - xL) / (fR - fL);
            if (!std::isfinite(xRF)) xRF = 0.5 * (xL + xR);
            double fRF = g(xRF);
            // maintain bracket
            if (fL * fRF <= 0.0) {
                xR = xRF;
                fR = fRF;
            } else {
                xL = xRF;
                fL = fRF;
            }
            // bisection safeguard
            double xBI = 0.5 * (xL + xR);
            double fBI = g(xBI);
            // choose the better next step
            if (std::abs(fRF) < std::abs(fBI)) {
                x = xRF;
                f = fRF;
            } else {
                x = xBI;
                f = fBI;
            }
        }
        return x;
    }
};

struct Bootstrapper {
    static shared_ptr<PWLogDF> build_discount_from_ois(const vector<OISSwap> &ois, const vector<double> &knots) {
        vector<double> rates(knots.size(), 0.0);
        for (size_t k = 1; k < knots.size(); ++k) {
            auto curve = make_shared<PWLogDF>(vector<double>(knots.begin(), knots.begin() + k + 1),
                                              vector<double>(rates.begin(), rates.begin() + k + 1));
            // pick OIS: nearest maturity to current knot (prefer >= knot if tie)
            const auto &curret_t = knots[k];
            const OISSwap *inst = nullptr;
            double best = 1e99;
            for (const auto &q: ois) {
                double T = q.fixed.times.back();
                double d = std::fabs(T - curret_t) - 1e-12 * (T >= curret_t ? 1.0 : 0.0);
                if (d < best) {
                    best = d;
                    inst = &q;
                }
            }
            if (!inst) throw runtime_error("no OIS near knot");
            auto f = [&](double x) {
                curve->set_last(x);
                return OISSwap::par_rate(*curve, inst->fixed);
            };
            auto zero = Root::bracketed_secant(f, -0.05, 1.0, inst->quoteRate);
            rates[k] = zero;
        }
        return make_shared<PWLogDF>(knots, rates);
    }

    static shared_ptr<PWLogForwardDF> build_forward_from_irs(const PWLogDF &D, const vector<IRSSwap> &irs,
                                                             const vector<double> &knots) {
        vector<double> rates(knots.size(), 0.0);
        for (size_t k = 1; k < knots.size(); ++k) {
            auto F = make_shared<PWLogForwardDF>(vector<double>(knots.begin(), knots.begin() + k + 1),
                                                 vector<double>(rates.begin(), rates.begin() + k + 1));
            auto inst = std::upper_bound(irs.begin(), irs.end(), knots[k],
                                         [](double t, const IRSSwap &i) { return t <= i.fixed.times.back(); });
            if (inst == irs.end()) throw runtime_error("no IRS");
            auto f = [&](double x) {
                F->set_last(x);
                return inst->par_rate(D, *F);
            };
            rates[k] = Root::bracketed_secant(f, -0.1, 0.5, inst->quoteRate);
        }
        return make_shared<PWLogForwardDF>(knots, rates);
    }

    static shared_ptr<PWLogForwardDF> build_forward_from_basis_and_irs(const PWLogDF &D, const PWLogForwardDF &Fanchor,
                                                                       const vector<IRSSwap> &irs,
                                                                       const vector<BasisSwap> &basis,
                                                                       const vector<double> &knots) {
        vector<double> logP(knots.size(), 0.0);
        for (size_t k = 1; k < knots.size(); ++k) {
            auto F = make_shared<PWLogForwardDF>(vector<double>(knots.begin(), knots.begin() + k + 1),
                                                 vector<double>(logP.begin(), logP.begin() + k + 1));
            const BasisSwap *bq = nullptr;
            double bestB = 1e99;
            for (auto &b: basis) {
                double T = max(b.leg1.times.back(), b.leg2.times.back());
                double d = fabs(T - knots[k]);
                if (T >= knots[k] && d < bestB) {
                    bq = &b;
                    bestB = d;
                }
            }
            if (!bq && !basis.empty()) {
                for (auto &b: basis) {
                    double T = max(b.leg1.times.back(), b.leg2.times.back());
                    double d = fabs(T - knots[k]);
                    if (d < bestB) {
                        bq = &b;
                        bestB = d;
                    }
                }
            }
            const IRSSwap *iq = nullptr;
            double bestI = 1e99;
            for (auto &q: irs) {
                double T = q.floatSched.times.back();
                double d = fabs(T - knots[k]);
                if (T >= knots[k] && d < bestI) {
                    iq = &q;
                    bestI = d;
                }
            }
            if (!iq && !irs.empty()) {
                for (auto &q: irs) {
                    double T = q.floatSched.times.back();
                    double d = fabs(T - knots[k]);
                    if (d < bestI) {
                        iq = &q;
                        bestI = d;
                    }
                }
            }
            if (!bq && !iq) throw runtime_error("no basis/irs");

            bool useBasis = (bq != nullptr);
            auto f = [&](double x) {
                F->set_last(x);
                if (useBasis) {
                    bool thisIsLeg1 = bq->leg1.times.size() <= bq->leg2.times.size();
                    if (thisIsLeg1) return bq->pv_diff(D, *F, Fanchor);
                    else return bq->pv_diff(D, Fanchor, *F);
                } else {
                    return iq->par_rate(D, *F);
                }
            };
            double target = useBasis ? 0.0 : iq->quoteRate;
            logP[k] = Root::bracketed_secant(f, -10.0, 0.5, target);
        }
        return make_shared<PWLogForwardDF>(knots, logP);
    }
};

/************* Market & model *************/
struct MarketData {
    vector<double> discKnots;
    vector<OISSwap> ois;
    vector<double> f3mKnots, f6mKnots;
    vector<IRSSwap> irs3m, irs6m;
    vector<BasisSwap> basis63;
};

struct ModelCurves {
    shared_ptr<PWLogDF> D;
    shared_ptr<PWLogForwardDF> F3M, F6M;
};

static ModelCurves build_all(const MarketData &m, const vector<double> &discNodes) {
    auto D = make_shared<PWLogDF>(m.discKnots, discNodes);
    auto F3 = Bootstrapper::build_forward_from_irs(*D, m.irs3m, m.f3mKnots);
    auto F6 = Bootstrapper::build_forward_from_basis_and_irs(*D, *F3, m.irs6m, m.basis63, m.f6mKnots);
    return {D, F3, F6};
}

/************* Residuals *************/
struct Residuals {
    vector<double> r;
    vector<string> label;
    vector<double> maturity;
};

static Residuals compute_residuals(const MarketData &m, const ModelCurves &mdl) {
    Residuals R;
    for (auto &q: m.ois) {
        double km = q.par_rate(*mdl.D);
        R.r.push_back(km - q.quoteRate);
        R.label.push_back("OIS");
        R.maturity.push_back(q.maturity);
    }
    for (auto &q: m.irs3m) {
        double T = q.floatSched.times.back();
        double km = q.par_rate(*mdl.D, *mdl.F3M);
        R.r.push_back(km - q.quoteRate);
        R.label.push_back("IRS3M");
        R.maturity.push_back(T);
    }
    for (auto &q: m.irs6m) {
        double T = q.floatSched.times.back();
        double km = q.par_rate(*mdl.D, *mdl.F6M);
        R.r.push_back(km - q.quoteRate);
        R.label.push_back("IRS6M");
        R.maturity.push_back(T);
    }
    for (auto &b: m.basis63) {
        double T = max(b.leg1.times.back(), b.leg2.times.back());
        double pv = b.pv_diff(*mdl.D, *mdl.F6M, *mdl.F3M);
        R.r.push_back(pv);
        R.label.push_back("BASIS63");
        R.maturity.push_back(T);
    }
    return R;
}

/************* Jacobians *************/
static pair<vector<vector<double> >, Residuals>
J_discount_analytic_direct(const MarketData &m, const ModelCurves &mdl) {
    Residuals R = compute_residuals(m, mdl);
    size_t mrows = R.r.size(), n = mdl.D->logDF.size();
    vector<vector<double> > J(mrows, vector<double>(n, 0.0));
    size_t row = 0;
    for (auto &q: m.ois) {
        auto d = q.d_par_rate_dtheta(*mdl.D);
        for (auto &kv: d) J[row][kv.first] += kv.second;
        row++;
    }
    for (auto &q: m.irs3m) {
        auto d = q.d_par_rate_dtheta(*mdl.D, *mdl.F3M);
        for (auto &kv: d) J[row][kv.first] += kv.second;
        row++;
    }
    for (auto &q: m.irs6m) {
        auto d = q.d_par_rate_dtheta(*mdl.D, *mdl.F6M);
        for (auto &kv: d) J[row][kv.first] += kv.second;
        row++;
    }
    for (auto &b: m.basis63) {
        auto d = b.d_pv_diff_dtheta(*mdl.D, *mdl.F6M, *mdl.F3M);
        for (auto &kv: d) J[row][kv.first] += kv.second;
        row++;
    }
    return {J, R};
}

// Jacobian wrt forward curve nodes (either 3M or 6M).
static pair<vector<vector<double> >, Residuals>
J_forward_analytic(const MarketData &m, const ModelCurves &mdl, bool threeM) {
    Residuals R = compute_residuals(m, mdl);
    const auto &F = threeM ? *mdl.F3M : *mdl.F6M;
    size_t mrows = R.r.size(), n = F.logDF.size();
    vector<vector<double> > J(mrows, vector<double>(n, 0.0));
    size_t row = 0;
    // OIS rows -> zero (depend only on discount)
    row += m.ois.size();
    if (threeM) {
        for (auto &q: m.irs3m) {
            auto d = q.d_par_rate_dphi(*mdl.D, *mdl.F3M);
            for (auto &kv: d) if ((size_t) kv.first < n) J[row][kv.first] += kv.second;
            row++;
        }
        // skip 6M IRS rows (no 3M forward sensitivity)
        row += m.irs6m.size();
        // Basis: pv(6M + s) - pv(3M) -> derivative w.r.t 3M is NEGATIVE term (implemented in which=2)
        for (auto &b: m.basis63) {
            auto d = b.d_pv_diff_dphi(*mdl.D, *mdl.F6M, *mdl.F3M, /*which=*/2);
            for (auto &kv: d) if ((size_t) kv.first < n) J[row][kv.first] += kv.second;
            row++;
        }
    } else {
        // 3M IRS rows (no 6M forward sensitivity)
        row += m.irs3m.size();
        // 6M IRS rows
        for (auto &q: m.irs6m) {
            auto d = q.d_par_rate_dphi(*mdl.D, *mdl.F6M);
            for (auto &kv: d) if ((size_t) kv.first < n) J[row][kv.first] += kv.second;
            row++;
        }
        // Basis: derivative w.r.t 6M is POSITIVE term (implemented in which=1)
        for (auto &b: m.basis63) {
            auto d = b.d_pv_diff_dphi(*mdl.D, *mdl.F6M, *mdl.F3M, /*which=*/1);
            for (auto &kv: d) if ((size_t) kv.first < n) J[row][kv.first] += kv.second;
            row++;
        }
    }
    return {J, R};
}

static vector<vector<double> > second_diff_matrix(int n) {
    if (n < 3) return vector<vector<double> >(0, vector<double>(n, 0.0));
    vector<vector<double> > L(n - 2, vector<double>(n, 0.0));
    for (int i = 0; i < n - 2; ++i) {
        L[i][i] = 1.0;
        L[i][i + 1] = -2.0;
        L[i][i + 2] = 1.0;
    }
    return L;
}

/************* Gauss–Newton with smoothness *************/
struct GNOptions {
    int max_iters = 15;
    double tol = 1e-12;
    double lambda_ridge = 1e-10;
    double lambda_smooth = 1e-6;
    bool verbose = true;
};

struct GNResult {
    vector<double> theta;
    double final_norm;
    int iters;
};

static GNResult gauss_newton_fit(const MarketData &m, const vector<double> &theta0, const GNOptions &opt) {
    vector<double> theta = theta0;
    int n = (int) theta.size();

    auto L = second_diff_matrix(n);

    auto Lt = [&](const vector<vector<double> > &A)-> vector<vector<double> > {
        int r = (int) A.size(), c = r ? (int) A[0].size() : 0;
        vector<vector<double> > T(c, vector<double>(r, 0.0));
        for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) T[j][i] = A[i][j];
        return T;
    };

    auto matmul = [&](const vector<vector<double> > &A, const vector<vector<double> > &B) {
        int r = (int) A.size();
        int k = r ? (int) A[0].size() : 0;
        int c = (int) B.size() ? (int) B[0].size() : 0;
        vector<vector<double> > C(r, vector<double>(c, 0.0));
        for (int i = 0; i < r; ++i) {
            for (int t = 0; t < k; ++t) {
                double a = A[i][t];
                if (a == 0) continue;
                for (int j = 0; j < c; ++j) C[i][j] += a * B[t][j];
            }
        }
        return C;
    };

    auto add_eye = [&](vector<vector<double> > &A, double lam) {
        for (int i = 1; i < (int) A.size(); ++i) if (i < (int) A[i].size()) A[i][i] += lam; // skip hard-pinned node 0
    };

    auto solve = [&](vector<vector<double> > A, vector<double> b) {
        int N = (int) A.size();
        for (int k = 0; k < N; ++k) {
            int piv = k;
            for (int i = k + 1; i < N; ++i) if (fabs(A[i][k]) > fabs(A[piv][k])) piv = i;
            if (fabs(A[piv][k]) < 1e-18) continue;
            if (piv != k) {
                swap(A[piv], A[k]);
                swap(b[piv], b[k]);
            }
            double diag = A[k][k];
            for (int j = k; j < N; ++j) A[k][j] /= diag;
            b[k] /= diag;
            for (int i = 0; i < N; ++i) {
                if (i == k) continue;
                double f = A[i][k];
                if (f == 0) continue;
                for (int j = k; j < N; ++j) A[i][j] -= f * A[k][j];
                b[i] -= f * b[k];
            }
        }
        return b;
    };

    for (int it = 0; it < opt.max_iters; ++it) {
        auto mdl = build_all(m, theta);
        auto R = compute_residuals(m, mdl);
        double nrm = 0;
        for (double v: R.r) nrm += v * v;
        if (opt.verbose) cerr << "[GN] iter " << it << " ||r||^2=" << setprecision(12) << nrm << "\n";

        // Jacobian wrt discount nodes
        auto JR = J_discount_analytic_direct(m, mdl);
        auto &J = JR.first;

        // Build normal equations
        vector<vector<double> > JTJ(n, vector<double>(n, 0.0));
        vector<double> g(n, 0.0);
        for (size_t i = 0; i < R.r.size(); ++i) {
            for (int j = 0; j < n; ++j) {
                g[j] += J[i][j] * R.r[i];
                for (int k = 0; k < n; ++k) JTJ[j][k] += J[i][j] * J[i][k];
            }
        }

        add_eye(JTJ, opt.lambda_ridge);

        // smoothness penalty: add lambda * (L^T L)
        vector<vector<double> > LtL;
        if (L.empty()) {
            LtL = vector<vector<double> >(n, vector<double>(n, 0.0));
        } else {
            auto LtM = Lt(L);
            LtL = matmul(LtM, L);
        }
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                JTJ[i][j] += opt.lambda_smooth * LtL[i][j];

        for (int j = 0; j < n; ++j) g[j] = -g[j];

        // constrain node 0 (hard pin)
        JTJ[0][0] += 1e12;
        g[0] = 0.0;

        auto dx = solve(JTJ, g);

        // line search
        double alpha = 1.0, best = nrm;
        vector<double> best_theta = theta;
        for (int ls = 0; ls < 6; ++ls) {
            vector<double> cand = theta;
            for (int j = 0; j < n; ++j) cand[j] += alpha * dx[j];
            auto mdl2 = build_all(m, cand);
            auto R2 = compute_residuals(m, mdl2);
            double n2 = 0;
            for (double v: R2.r) n2 += v * v;
            if (n2 < best) {
                best = n2;
                best_theta = cand;
            }
            alpha *= 0.5;
        }
        if (fabs(best - nrm) < 1e-14) return {best_theta, best, it + 1};
        theta = best_theta;
    }
    auto mdl = build_all(m, theta);
    auto R = compute_residuals(m, mdl);
    double nrm = 0;
    for (double v: R.r) nrm += v * v;
    return {theta, nrm, 999};
}

/************* Demo *************/
int main() {
    try {
        DayCount dcOIS = DayCount::ACT365F,
                dcFixed = DayCount::ACT365F,
                dc3M = DayCount::ACT360,
                dc6M = DayCount::ACT360;

        int fixedPY = 1,
                py3M = 4,
                py6M = 2;

        MarketData m;
        m.discKnots = {0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};

        auto addOIS = [&](double TMat, double K) {
            m.ois.push_back({Schedule::regular(TMat, fixedPY, dcFixed), TMat, K, dcFixed});
        };
        addOIS(0.25, 0.0260);
        addOIS(0.50, 0.0265);
        addOIS(1.00, 0.0270);
        addOIS(2.00, 0.0285);
        addOIS(3.00, 0.0295);
        addOIS(5.00, 0.0310);
        addOIS(7.00, 0.0315);
        addOIS(10.0, 0.0320);

        m.f3mKnots = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0};
        m.f6mKnots = {0.0, 2.0, 3.0, 5.0, 7.0, 10.0};

        auto addIRS3 = [&](double T, double K) {
            m.irs3m.push_back({Schedule::regular(T, fixedPY, dcFixed), Schedule::regular(T, py3M, dc3M), K, dcFixed});
        };
        addIRS3(0.5, 0.0230);
        addIRS3(1.0, 0.0335);
        addIRS3(2.0, 0.0340);
        addIRS3(3.0, 0.0362);
        addIRS3(5.0, 0.0375);
        addIRS3(7.0, 0.0407);
        addIRS3(10.0, 0.0500);

        auto addIRS6 = [&](double T, double K) {
            m.irs6m.push_back({Schedule::regular(T, fixedPY, dcFixed), Schedule::regular(T, py6M, dc6M), K, dcFixed});
        };
        addIRS6(2.0, 0.0348);
        addIRS6(5.0, 0.0351);
        addIRS6(10.0, 0.0356);

        auto addBasis = [&](double T, double b) {
            m.basis63.push_back({Schedule::regular(T, py6M, dc6M), Schedule::regular(T, py3M, dc3M), b});
        };
        addBasis(2.0, 0.0008);
        addBasis(3.0, 0.0009);
        addBasis(5.0, 0.0010);
        addBasis(7.0, 0.0011);
        addBasis(10.0, 0.0012);

        // Initial theta from OIS bootstrap
        auto D0 = Bootstrapper::build_discount_from_ois(m.ois, m.discKnots);
        vector<double> theta0 = D0->rates;

        GNOptions opt;
        opt.max_iters = 10;
        opt.lambda_ridge = 1e-10;
        opt.lambda_smooth = 1e-6;
        opt.verbose = true;

        auto gn = gauss_newton_fit(m, theta0, opt);
        cerr << "Finished GN. ||r||^2=" << gn.final_norm << " iters=" << gn.iters << "\n";

        auto mdl = build_all(m, gn.theta);

        // Print repricing
        cout.setf(std::ios::fixed);
        cout << setprecision(6);
        cout << "OIS repricing:\n";
        for (auto &q: m.ois) {
            double km = q.par_rate(*mdl.D);
            cout << q.maturity << "y " << km << " vs " << q.quoteRate << " diff=" << (km - q.quoteRate) << "\n";
        }
        cout << "\nIRS3M:\n";
        for (auto &q: m.irs3m) {
            double T = q.floatSched.times.back();
            double km = q.par_rate(*mdl.D, *mdl.F3M);
            cout << T << "y " << km << " vs " << q.quoteRate << " diff=" << (km - q.quoteRate) << "\n";
        }
        cout << "\nIRS6M:\n";
        for (auto &q: m.irs6m) {
            double T = q.floatSched.times.back();
            double km = q.par_rate(*mdl.D, *mdl.F6M);
            cout << T << "y " << km << " vs " << q.quoteRate << " diff=" << (km - q.quoteRate) << "\n";
        }
        cout << "\nBasis (6M+b vs 3M) PV-diff:\n";
        for (auto &b: m.basis63) {
            double T = max(b.leg1.times.back(), b.leg2.times.back());
            double pv = b.pv_diff(*mdl.D, *mdl.F6M, *mdl.F3M);
            cout << T << "y " << pv << "\n";
        }

        // Discount bucketed risk (analytic direct)
        auto JR = J_discount_analytic_direct(m, mdl);
        auto &J = JR.first;
        auto R = JR.second;
        cout << "\nBucketed risk (discount logDF nodes):\n";
        for (size_t i = 0; i < J.size(); ++i) {
            cout << setw(12) << R.label[i] << "  ";
            for (size_t j = 0; j < mdl.D->logDF.size(); ++j) cout << setw(12) << J[i][j];
            cout << "\n";
        }

        // Forward bucketed risk (3M)
        auto JF3_pair = J_forward_analytic(m, mdl, /*threeM=*/true);
        const auto &JF3 = JF3_pair.first;
        const auto &R3 = JF3_pair.second;
        cout << "\nBucketed risk (3M forward logDF nodes):\n";
        for (size_t i = 0; i < JF3.size(); ++i) {
            cout << setw(12) << R3.label[i] << "  ";
            for (size_t j = 0; j < mdl.F3M->logDF.size(); ++j) cout << setw(12) << JF3[i][j];
            cout << "\n";
        }

        // Forward bucketed risk (6M)
        auto JF6_pair = J_forward_analytic(m, mdl, /*threeM=*/false);
        const auto &JF6 = JF6_pair.first;
        const auto &R6 = JF6_pair.second;
        cout << "\nBucketed risk (6M forward logDF nodes):\n";
        for (size_t i = 0; i < JF6.size(); ++i) {
            cout << setw(12) << R6.label[i] << "  ";
            for (size_t j = 0; j < mdl.F6M->logDF.size(); ++j) cout << setw(12) << JF6[i][j];
            cout << "\n";
        }

        return 0;
    } catch (const exception &e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
