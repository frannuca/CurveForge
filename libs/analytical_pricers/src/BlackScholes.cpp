//
// Created by Francisco Nunez on 29.11.2025.
//
#include "analytical_pricers/BlackScholes.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace curve::analytical_pricers {
    double BlackScholes::norm_cdf(double x) {
        // Approximation using error function
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

    double BlackScholes::norm_pdf(double x) {
        return std::exp(-0.5 * x * x) / SQRT_2PI;
    }

    double BlackScholes::d1(double S, double K, double r, double sigma, double T) {
        if (T <= 0.0 || sigma <= 0.0) {
            throw std::invalid_argument("Time to maturity and volatility must be positive");
        }
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    double BlackScholes::d2(double S, double K, double r, double sigma, double T) {
        return d1(S, K, r, sigma, T) - sigma * std::sqrt(T);
    }

    double BlackScholes::call_price(double S, double K, double r, double sigma, double T) {
        if (T <= 0.0) return std::max(S - K, 0.0);
        if (sigma <= 0.0) return std::max(S - K * std::exp(-r * T), 0.0);

        double d1_val = d1(S, K, r, sigma, T);
        double d2_val = d2(S, K, r, sigma, T);

        return S * norm_cdf(d1_val) - K * std::exp(-r * T) * norm_cdf(d2_val);
    }

    double BlackScholes::put_price(double S, double K, double r, double sigma, double T) {
        if (T <= 0.0) return std::max(K - S, 0.0);
        if (sigma <= 0.0) return std::max(K * std::exp(-r * T) - S, 0.0);

        double d1_val = d1(S, K, r, sigma, T);
        double d2_val = d2(S, K, r, sigma, T);

        return K * std::exp(-r * T) * norm_cdf(-d2_val) - S * norm_cdf(-d1_val);
    }

    double BlackScholes::vega(double S, double K, double r, double sigma, double T) {
        if (T <= 0.0 || sigma <= 0.0) return 0.0;

        double d1_val = d1(S, K, r, sigma, T);
        return S * norm_pdf(d1_val) * std::sqrt(T);
    }

    double BlackScholes::implied_volatility(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        bool is_call,
        double initial_guess,
        double tolerance,
        int max_iterations
    ) {
        if (market_price <= 0.0) {
            throw std::invalid_argument("Market price must be positive");
        }

        double sigma = std::max(MIN_VOL, std::min(MAX_VOL, initial_guess));

        for (int i = 0; i < max_iterations; ++i) {
            double price = is_call
                               ? call_price(S, K, r, sigma, T)
                               : put_price(S, K, r, sigma, T);
            double diff = price - market_price;

            if (std::abs(diff) < tolerance) {
                return sigma;
            }

            double vega_val = vega(S, K, r, sigma, T);
            if (vega_val < 1e-10) {
                // Vega too small, switch to Brent's method
                return implied_volatility_brent(market_price, S, K, r, T, is_call);
            }
            sigma = sigma - diff / vega_val;
            sigma = std::max(MIN_VOL, std::min(MAX_VOL, sigma));
        }

        throw std::runtime_error("Implied volatility did not converge");
    }

    double BlackScholes::implied_volatility_brent(
        double market_price,
        double S,
        double K,
        double r,
        double T,
        bool is_call,
        double vol_min,
        double vol_max,
        double tolerance,
        int max_iterations
    ) {
        auto price_func = [&](double sigma) {
            double price = is_call
                               ? call_price(S, K, r, sigma, T)
                               : put_price(S, K, r, sigma, T);
            return price - market_price;
        };

        double a = vol_min;
        double b = vol_max;
        double fa = price_func(a);
        double fb = price_func(b);

        if (fa * fb > 0) {
            throw std::runtime_error("Brent's method: root not bracketed");
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        double c = a;
        double fc = fa;
        bool mflag = true;
        double s = 0.0;
        double d = 0.0;

        for (int i = 0; i < max_iterations; ++i) {
            if (std::abs(b - a) < tolerance) {
                return b;
            }

            if (fa != fc && fb != fc) {
                // Inverse quadratic interpolation
                s = a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb));
            } else {
                // Secant method
                s = b - fb * (b - a) / (fb - fa);
            }

            double tmp2 = (3 * a + b) / 4;
            bool cond1 = !((s > tmp2 && s < b) || (s < tmp2 && s > b));
            bool cond2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2;
            bool cond3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2;
            bool cond4 = mflag && std::abs(b - c) < tolerance;
            bool cond5 = !mflag && std::abs(c - d) < tolerance;

            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2;
                mflag = true;
            } else {
                mflag = false;
            }

            double fs = price_func(s);
            d = c;
            c = b;
            fc = fb;

            if (fa * fs < 0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }

            if (std::abs(fa) < std::abs(fb)) {
                std::swap(a, b);
                std::swap(fa, fb);
            }
        }

        throw std::runtime_error("Brent's method did not converge");
    }
} // namespace curve::volatility
//
// Created by Francisco Nunez on 29.11.2025.
//

#ifndef CURVEFORGE_BLACKSCHOLES_H
#define CURVEFORGE_BLACKSCHOLES_H


#include <cmath>
#include <stdexcept>

namespace curve::volatility {
    /**
     * @brief Black-Scholes option pricing and implied volatility utilities
     */
    class BlackScholes {
    public:
        /**
         * @brief Standard normal cumulative distribution function
         */
        static double norm_cdf(double x);

        /**
         * @brief Standard normal probability density function
         */
        static double norm_pdf(double x);

        /**
         * @brief Calculate d1 parameter in Black-Scholes formula
         */
        static double d1(double S, double K, double r, double sigma, double T);

        /**
         * @brief Calculate d2 parameter in Black-Scholes formula
         */
        static double d2(double S, double K, double r, double sigma, double T);

        /**
         * @brief Black-Scholes European call option price
         * @param S Current spot price
         * @param K Strike price
         * @param r Risk-free rate
         * @param sigma Volatility
         * @param T Time to maturity (years)
         * @return Call option price
         */
        static double call_price(double S, double K, double r, double sigma, double T);

        /**
         * @brief Black-Scholes European put option price
         * @param S Current spot price
         * @param K Strike price
         * @param r Risk-free rate
         * @param sigma Volatility
         * @param T Time to maturity (years)
         * @return Put option price
         */
        static double put_price(double S, double K, double r, double sigma, double T);

        /**
         * @brief Calculate option vega (sensitivity to volatility)
         */
        static double vega(double S, double K, double r, double sigma, double T);

        /**
         * @brief Calculate implied volatility using Newton-Raphson method
         * @param market_price Observed market price
         * @param S Current spot price
         * @param K Strike price
         * @param r Risk-free rate
         * @param T Time to maturity (years)
         * @param is_call True for call, false for put
         * @param initial_guess Initial volatility guess
         * @param tolerance Convergence tolerance
         * @param max_iterations Maximum number of iterations
         * @return Implied volatility
         */
        static double implied_volatility(
            double market_price,
            double S,
            double K,
            double r,
            double T,
            bool is_call = true,
            double initial_guess = 0.3,
            double tolerance = 1e-6,
            int max_iterations = 100
        );

        /**
         * @brief Calculate implied volatility using Brent's method (more robust)
         */
        static double implied_volatility_brent(
            double market_price,
            double S,
            double K,
            double r,
            double T,
            bool is_call = true,
            double vol_min = 0.001,
            double vol_max = 5.0,
            double tolerance = 1e-6,
            int max_iterations = 100
        );

    private:
        static constexpr double SQRT_2PI = 2.506628274631000502;
        static constexpr double MIN_VOL = 1e-4;
        static constexpr double MAX_VOL = 10.0;
    };
} // namespace curve::volatility

#endif //CURVEFORGE_BLACKSCHOLES_H
