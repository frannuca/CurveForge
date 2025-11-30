//
// Created by Francisco Nunez on 29.11.2025.
//

#ifndef CURVEFORGE_IMPLIEDVOLSURFACE_H
#define CURVEFORGE_IMPLIEDVOLSURFACE_H

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "datacontracts/vol.hxx"
#include "datacontracts/marketdata.hxx"
#include "interpolation/bspline.h"
#include "OptionQuote.h"
#include "VolPoint.h"

namespace curve::volatility {
    /**
     * @brief Implied Volatility Surface Calibration
     *
     * Calibrates an implied volatility surface from option market prices.
     * Supports multiple interpolation methods and parametrizations.
     */
    class ImpliedVolSurface {
    public:
        enum class SurfaceType {
            STRIKE_SPACE, // Interpolate in (K, T) space
            MONEYNESS_SPACE, // Interpolate in (K/F, T) space
            LOG_MONEYNESS_SPACE // Interpolate in (ln(K/F), T) space
        };

        enum class InterpolationMethod {
            BILINEAR, // Bilinear interpolation
            BICUBIC_SPLINE, // Bicubic spline interpolation
            LINEAR_IN_VARIANCE // Linear in total variance
        };

        /**
         * @brief Constructor
         * @param surface_type Type of surface parametrization
         * @param interp_method Interpolation method
         * @param risk_free_rate Risk-free rate (can be curve later)
         */
        ImpliedVolSurface(
            SurfaceType surface_type = SurfaceType::LOG_MONEYNESS_SPACE,
            InterpolationMethod interp_method = InterpolationMethod::BICUBIC_SPLINE,
            double risk_free_rate = 0.0
        );

        /**
         * @brief Calibrate the surface from option quotes
         * @param quotes Vector of option market quotes
         * @return True if calibration successful
         */
        bool calibrate(const std::vector<OptionQuote> &quotes);

        /**
         * @brief Get implied volatility at a given strike and maturity
         * @param strike Strike price
         * @param maturity Time to maturity (years)
         * @param forward Forward price
         * @return Implied volatility
         */
        double get_volatility(double strike, double maturity, double forward) const;

        /**
         * @brief Get implied volatility at moneyness and maturity
         * @param moneyness Moneyness (K/F or ln(K/F))
         * @param maturity Time to maturity (years)
         * @return Implied volatility
         */
        double get_volatility_by_moneyness(double moneyness, double maturity) const;

        /**
         * @brief Export calibrated surface to XSD VolSurface format
         * @param underlying_id Underlying identifier
         * @param as_of As-of date
         * @return VolSurface object
         */
        std::unique_ptr<vol::VolSurface> export_to_vol_surface(
            const std::string &underlying_id,
            const xml_schema::date &as_of
        ) const;

        /**
         * @brief Import from XSD VolSurface format
         */
        bool import_from_vol_surface(const vol::VolSurface &surface);

        /**
         * @brief Get all calibrated volatility points
         */
        const std::vector<VolPoint> &get_calibrated_points() const { return calibrated_points_; }

        /**
         * @brief Get calibration statistics
         */
        struct CalibrationStats {
            double mean_error;
            double max_error;
            double rmse;
            int num_points;
        };

        CalibrationStats get_calibration_stats(const std::vector<OptionQuote> &quotes) const;

    private:
        SurfaceType surface_type_;
        InterpolationMethod interp_method_;
        double risk_free_rate_;

        std::vector<VolPoint> calibrated_points_;

        // For spline interpolation
        std::unique_ptr<interpolation::bspline> maturity_splines_;
        std::map<double, std::unique_ptr<interpolation::bspline> > strike_splines_;

        // Grid data for bilinear interpolation
        std::vector<double> maturity_grid_;
        std::vector<double> strike_grid_;
        Eigen::MatrixXd vol_grid_;

        // Helper methods
        double compute_moneyness(double strike, double forward) const;

        double interpolate_bilinear(double x, double y) const;

        double interpolate_bicubic(double x, double y) const;

        void build_interpolation_grid();

        void validate_no_arbitrage() const;
    };

    /**
     * @brief Factory for creating ImpliedVolSurface from market data
     */
    class ImpliedVolSurfaceFactory {
    public:
        /**
         * @brief Calibrate surface from market data snapshot
         * @param md Market data snapshot containing option quotes
         * @param underlying_id Underlying identifier
         * @return Calibrated volatility surface
         */
        static std::unique_ptr<vol::VolSurface> calibrate_from_market_data(
            const marketdata::MarketDataSnapshot &md,
            const std::string &underlying_id
        );

        /**
         * @brief Extract option quotes from market data
         */
        static std::vector<OptionQuote> extract_option_quotes(
            const marketdata::MarketDataSnapshot &md,
            const std::string &underlying_id
        );
    };
} // namespace curve::volatility

#endif //CURVEFORGE_IMPLIEDVOLSURFACE_H
