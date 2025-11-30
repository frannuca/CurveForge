//
// Created by Francisco Nunez on 29.11.2025.
//

#include "volatility/ImpliedVolSurface.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <set>

#include "analytical_pricers/BlackScholes.h"

namespace curve::volatility {
    using namespace curve::analytical_pricers;

    ImpliedVolSurface::ImpliedVolSurface(
        SurfaceType surface_type,
        InterpolationMethod interp_method,
        double risk_free_rate
    ) : surface_type_(surface_type),
        interp_method_(interp_method),
        risk_free_rate_(risk_free_rate) {
    }

    bool ImpliedVolSurface::calibrate(const std::vector<OptionQuote> &quotes) {
        if (quotes.empty()) {
            return false;
        }

        calibrated_points_.clear();

        // Step 1: Compute implied volatility for each quote
        for (const auto &quote: quotes) {
            try {
                double forward = quote.forward > 0 ? quote.forward : quote.spot;
                double implied_vol = BlackScholes::implied_volatility(
                    quote.market_price,
                    quote.spot,
                    quote.strike,
                    risk_free_rate_,
                    quote.maturity,
                    quote.is_call,
                    0.3, // initial guess
                    1e-6,
                    100
                );

                double moneyness = compute_moneyness(quote.strike, forward);
                calibrated_points_.emplace_back(quote.strike, quote.maturity, implied_vol, moneyness);
            } catch (const std::exception &e) {
                // Skip quotes that fail to calibrate
                continue;
            }
        }

        if (calibrated_points_.empty()) {
            return false;
        }

        // Step 2: Build interpolation structure
        build_interpolation_grid();

        return true;
    }

    double ImpliedVolSurface::compute_moneyness(double strike, double forward) const {
        if (forward <= 0) {
            throw std::invalid_argument("Forward price must be positive");
        }

        switch (surface_type_) {
            case SurfaceType::STRIKE_SPACE:
                return strike;
            case SurfaceType::MONEYNESS_SPACE:
                return strike / forward;
            case SurfaceType::LOG_MONEYNESS_SPACE:
                return std::log(strike / forward);
            default:
                return strike;
        }
    }

    void ImpliedVolSurface::build_interpolation_grid() {
        if (calibrated_points_.empty()) {
            return;
        }

        // Extract unique maturities and strikes/moneyness
        std::set<double> maturity_set;
        std::set<double> strike_set;

        for (const auto &point: calibrated_points_) {
            maturity_set.insert(point.maturity);
            strike_set.insert(surface_type_ == SurfaceType::STRIKE_SPACE ? point.strike : point.moneyness);
        }

        maturity_grid_.assign(maturity_set.begin(), maturity_set.end());
        strike_grid_.assign(strike_set.begin(), strike_set.end());

        // Build volatility grid
        vol_grid_ = Eigen::MatrixXd::Zero(strike_grid_.size(), maturity_grid_.size());

        // Fill the grid with calibrated values
        std::map<std::pair<double, double>, double> point_map;
        for (const auto &point: calibrated_points_) {
            double x = surface_type_ == SurfaceType::STRIKE_SPACE ? point.strike : point.moneyness;
            point_map[{x, point.maturity}] = point.volatility;
        }

        for (size_t i = 0; i < strike_grid_.size(); ++i) {
            for (size_t j = 0; j < maturity_grid_.size(); ++j) {
                auto key = std::make_pair(strike_grid_[i], maturity_grid_[j]);
                if (point_map.count(key)) {
                    vol_grid_(i, j) = point_map[key];
                }
            }
        }

        // For spline interpolation, build per-maturity splines
        if (interp_method_ == InterpolationMethod::BICUBIC_SPLINE) {
            for (size_t j = 0; j < maturity_grid_.size(); ++j) {
                std::vector<Eigen::VectorXd> control_points;
                for (size_t i = 0; i < strike_grid_.size(); ++i) {
                    Eigen::VectorXd point(1);
                    point(0) = vol_grid_(i, j);
                    control_points.push_back(point);
                }
                if (control_points.size() >= 2) {
                    strike_splines_[maturity_grid_[j]] =
                            interpolation::bspline::interpolate(control_points,
                                                                std::min(3UL, control_points.size() - 1));
                }
            }
        }
    }

    double ImpliedVolSurface::get_volatility(double strike, double maturity, double forward) const {
        double moneyness = compute_moneyness(strike, forward);
        return get_volatility_by_moneyness(moneyness, maturity);
    }

    double ImpliedVolSurface::get_volatility_by_moneyness(double moneyness, double maturity) const {
        if (calibrated_points_.empty()) {
            throw std::runtime_error("Surface not calibrated");
        }

        // TODO: we do not track strikes at the moment to extract strikes.
        double x = surface_type_ == SurfaceType::STRIKE_SPACE ? moneyness : moneyness;

        switch (interp_method_) {
            case InterpolationMethod::BILINEAR:
                return interpolate_bilinear(x, maturity);
            case InterpolationMethod::BICUBIC_SPLINE:
                return interpolate_bicubic(x, maturity);
            case InterpolationMethod::LINEAR_IN_VARIANCE: {
                // Interpolate variance, then convert back to vol
                double variance = interpolate_bilinear(x, maturity);
                return std::sqrt(std::max(0.0, variance * variance));
            }
            default:
                return interpolate_bilinear(x, maturity);
        }
    }

    double ImpliedVolSurface::interpolate_bilinear(double x, double y) const {
        // Find surrounding grid points
        auto x_it = std::lower_bound(strike_grid_.begin(), strike_grid_.end(), x);
        auto y_it = std::lower_bound(maturity_grid_.begin(), maturity_grid_.end(), y);

        // Handle boundary cases
        if (x_it == strike_grid_.begin()) x_it++;
        if (y_it == maturity_grid_.begin()) y_it++;
        if (x_it == strike_grid_.end()) x_it = strike_grid_.end() - 1;
        if (y_it == maturity_grid_.end()) y_it = maturity_grid_.end() - 1;

        size_t i1 = std::distance(strike_grid_.begin(), x_it) - 1;
        size_t i2 = i1 + 1;
        size_t j1 = std::distance(maturity_grid_.begin(), y_it) - 1;
        size_t j2 = j1 + 1;

        if (i2 >= strike_grid_.size()) i2 = strike_grid_.size() - 1;
        if (j2 >= maturity_grid_.size()) j2 = maturity_grid_.size() - 1;

        double x1 = strike_grid_[i1];
        double x2 = strike_grid_[i2];
        double y1 = maturity_grid_[j1];
        double y2 = maturity_grid_[j2];

        if (x2 == x1 || y2 == y1) {
            // Return nearest value
            return vol_grid_(i1, j1);
        }

        // Bilinear interpolation
        double tx = (x - x1) / (x2 - x1);
        double ty = (y - y1) / (y2 - y1);

        double v11 = vol_grid_(i1, j1);
        double v12 = vol_grid_(i1, j2);
        double v21 = vol_grid_(i2, j1);
        double v22 = vol_grid_(i2, j2);

        return (1 - tx) * (1 - ty) * v11
               + (1 - tx) * ty * v12
               + tx * (1 - ty) * v21
               + tx * ty * v22;
    }

    double ImpliedVolSurface::interpolate_bicubic(double x, double y) const {
        // Find the two closest maturities
        auto y_it = std::lower_bound(maturity_grid_.begin(), maturity_grid_.end(), y);

        if (y_it == maturity_grid_.begin()) {
            y_it++;
        }
        if (y_it == maturity_grid_.end()) {
            y_it = maturity_grid_.end() - 1;
        }

        size_t j1 = std::distance(maturity_grid_.begin(), y_it) - 1;
        size_t j2 = j1 + 1;

        if (j2 >= maturity_grid_.size()) {
            j2 = maturity_grid_.size() - 1;
        }

        double y1 = maturity_grid_[j1];
        double y2 = maturity_grid_[j2];

        // Evaluate splines at both maturities
        double vol1 = 0.0, vol2 = 0.0;

        if (strike_splines_.count(y1)) {
            // Map x to [0,1] for spline evaluation
            double u = (x - strike_grid_.front()) / (strike_grid_.back() - strike_grid_.front());
            u = std::max(0.0, std::min(1.0, u));
            auto result = strike_splines_.at(y1)->evaluate(u);
            vol1 = result(0);
        }

        if (strike_splines_.count(y2)) {
            double u = (x - strike_grid_.front()) / (strike_grid_.back() - strike_grid_.front());
            u = std::max(0.0, std::min(1.0, u));
            auto result = strike_splines_.at(y2)->evaluate(u);
            vol2 = result(0);
        }

        // Linear interpolation in maturity dimension
        if (y2 == y1) {
            return vol1;
        }
        double ty = (y - y1) / (y2 - y1);
        return (1 - ty) * vol1 + ty * vol2;
    }

    std::unique_ptr<vol::VolSurface> ImpliedVolSurface::export_to_vol_surface(
        const std::string &underlying_id,
        const xml_schema::date &as_of
    ) const {
        // Create header with all required parameters
        vol::VolSurfaceHeader header(
            underlying_id,
            as_of,
            vol::VolQuoteType::BLACK,
            vol::StrikeDimensionType::ABSOLUTE_STRIKE // Default, will be updated below
        );

        // Set strike dimension based on surface type
        if (surface_type_ == SurfaceType::STRIKE_SPACE) {
            header.strikeDimension(vol::StrikeDimensionType::ABSOLUTE_STRIKE);
        } else if (surface_type_ == SurfaceType::MONEYNESS_SPACE ||
                   surface_type_ == SurfaceType::LOG_MONEYNESS_SPACE) {
            header.strikeDimension(vol::StrikeDimensionType::MONEYNESS);
        }

        // Set interpolation hints
        header.expiryInterpolation(vol::InterpolationType::LINEAR);
        if (interp_method_ == InterpolationMethod::BICUBIC_SPLINE) {
            header.strikeInterpolation(vol::InterpolationType::CUBIC_SPLINE);
        } else {
            header.strikeInterpolation(vol::InterpolationType::LINEAR);
        }

        // Create points
        vol::VolSurface::points_type points;

        for (const auto &point: calibrated_points_) {
            // Convert maturity to tenor string (simplified)
            int years = static_cast<int>(point.maturity);
            int months = static_cast<int>((point.maturity - years) * 12);

            std::string tenor;
            if (years > 0) {
                tenor = std::to_string(years) + "Y";
            } else if (months > 0) {
                tenor = std::to_string(months) + "M";
            } else {
                int days = static_cast<int>(point.maturity * 365);
                tenor = std::to_string(days) + "D";
            }

            double strike_coord = surface_type_ == SurfaceType::STRIKE_SPACE ? point.strike : point.moneyness;

            vol::VolSurfacePoint vol_point(
                vol::ExpiryTenor(tenor),
                strike_coord,
                point.volatility
            );

            points.point().push_back(vol_point);
        }

        return std::make_unique<vol::VolSurface>(header, points);
    }

    bool ImpliedVolSurface::import_from_vol_surface(const vol::VolSurface &surface) {
        calibrated_points_.clear();

        const auto &points = surface.points().point();

        for (const auto &point: points) {
            // Parse tenor to years (simplified)
            std::string tenor_str = point.expiry();
            double maturity = 0.0;

            if (tenor_str.back() == 'Y') {
                maturity = std::stod(tenor_str.substr(0, tenor_str.size() - 1));
            } else if (tenor_str.back() == 'M') {
                maturity = std::stod(tenor_str.substr(0, tenor_str.size() - 1)) / 12.0;
            } else if (tenor_str.back() == 'D') {
                maturity = std::stod(tenor_str.substr(0, tenor_str.size() - 1)) / 365.0;
            }

            double strike_coord = point.strikeCoordinate();
            double volatility = point.volatility();

            calibrated_points_.emplace_back(strike_coord, maturity, volatility, strike_coord);
        }

        build_interpolation_grid();
        return !calibrated_points_.empty();
    }

    ImpliedVolSurface::CalibrationStats ImpliedVolSurface::get_calibration_stats(
        const std::vector<OptionQuote> &quotes
    ) const {
        CalibrationStats stats{0.0, 0.0, 0.0, 0};

        if (quotes.empty()) {
            return stats;
        }

        double sum_error = 0.0;
        double sum_squared_error = 0.0;
        double max_error = 0.0;

        for (const auto &quote: quotes) {
            try {
                double forward = quote.forward > 0 ? quote.forward : quote.spot;
                double calibrated_vol = get_volatility(quote.strike, quote.maturity, forward);

                double model_price = quote.is_call
                                         ? BlackScholes::call_price(quote.spot, quote.strike, risk_free_rate_,
                                                                    calibrated_vol, quote.maturity)
                                         : BlackScholes::put_price(quote.spot, quote.strike, risk_free_rate_,
                                                                   calibrated_vol, quote.maturity);

                double error = std::abs(model_price - quote.market_price);
                sum_error += error;
                sum_squared_error += error * error;
                max_error = std::max(max_error, error);
                stats.num_points++;
            } catch (...) {
                continue;
            }
        }

        if (stats.num_points > 0) {
            stats.mean_error = sum_error / stats.num_points;
            stats.rmse = std::sqrt(sum_squared_error / stats.num_points);
            stats.max_error = max_error;
        }

        return stats;
    }

    void ImpliedVolSurface::validate_no_arbitrage() const {
        // TODO: Implement no-arbitrage checks
        // - Butterfly arbitrage
        // - Calendar arbitrage
    }

    // ImpliedVolSurfaceFactory implementation

    std::unique_ptr<vol::VolSurface> ImpliedVolSurfaceFactory::calibrate_from_market_data(
        const marketdata::MarketDataSnapshot &md,
        const std::string &underlying_id
    ) {
        // Extract option quotes
        auto quotes = extract_option_quotes(md, underlying_id);

        if (quotes.empty()) {
            throw std::runtime_error("No option quotes found for underlying: " + underlying_id);
        }

        // Create and calibrate surface
        ImpliedVolSurface surface(
            ImpliedVolSurface::SurfaceType::LOG_MONEYNESS_SPACE,
            ImpliedVolSurface::InterpolationMethod::BICUBIC_SPLINE,
            0.0 // TODO: Extract risk-free rate from market data
        );

        if (!surface.calibrate(quotes)) {
            throw std::runtime_error("Failed to calibrate volatility surface");
        }

        // Export to VolSurface format
        auto as_of = md.header().asOf();
        return surface.export_to_vol_surface(underlying_id, as_of);
    }

    std::vector<OptionQuote> ImpliedVolSurfaceFactory::extract_option_quotes(
        const marketdata::MarketDataSnapshot &md,
        const std::string &underlying_id
    ) {
        std::vector<OptionQuote> quotes;

        // TODO: Extract from quote sets in market data
        // This requires proper implementation once the quote schema is fully defined

        return quotes;
    }
} // namespace curve::volatility
