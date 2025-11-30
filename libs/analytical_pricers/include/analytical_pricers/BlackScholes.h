//
// Created by Francisco Nunez on 29.11.2025.
//

#ifndef CURVEFORGE_BLACKSCHOLES_H
#define CURVEFORGE_BLACKSCHOLES_H

#include <cmath>
#include <stdexcept>

namespace curve::analytical_pricers {
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

