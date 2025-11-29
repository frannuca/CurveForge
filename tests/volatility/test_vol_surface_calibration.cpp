//
// Volatility Surface Calibration Example
// Demonstrates how to calibrate an implied volatility surface from option prices
//

#include <iostream>
#include <iomanip>
#include <vector>
#include "volatility/ImpliedVolSurface.h"
#include "volatility/BlackScholes.h"

using namespace curve::volatility;

void print_separator() {
    std::cout << std::string(80, '=') << std::endl;
}

void example_basic_calibration() {
    std::cout << "\n";
    print_separator();
    std::cout << "EXAMPLE 1: Basic Volatility Surface Calibration" << std::endl;
    print_separator();

    // Create synthetic option market data
    double spot = 100.0;
    double risk_free_rate = 0.05;

    std::vector<OptionQuote> market_quotes;

    // Create a volatility smile across different strikes and maturities
    std::vector<double> strikes = {80, 90, 95, 100, 105, 110, 120};
    std::vector<double> maturities = {0.25, 0.5, 1.0, 2.0};

    std::cout << "\nGenerating synthetic option prices with volatility smile...\n";
    std::cout << std::left << std::setw(10) << "Strike"
            << std::setw(12) << "Maturity"
            << std::setw(12) << "True Vol"
            << std::setw(12) << "Price" << std::endl;
    std::cout << std::string(46, '-') << std::endl;

    for (double maturity: maturities) {
        for (double strike: strikes) {
            // Create a volatility smile: higher vol for OTM options
            double moneyness = std::log(strike / spot);
            double true_vol = 0.20 + 0.15 * moneyness * moneyness; // Smile shape

            // Generate market price using Black-Scholes
            bool is_call = (strike >= spot);
            double market_price = is_call
                                      ? BlackScholes::call_price(spot, strike, risk_free_rate, true_vol, maturity)
                                      : BlackScholes::put_price(spot, strike, risk_free_rate, true_vol, maturity);

            OptionQuote quote;
            quote.strike = strike;
            quote.maturity = maturity;
            quote.market_price = market_price;
            quote.spot = spot;
            quote.forward = spot * std::exp(risk_free_rate * maturity);
            quote.is_call = is_call;

            market_quotes.push_back(quote);

            if (maturity == 1.0) {
                // Print only 1Y for brevity
                std::cout << std::fixed << std::setprecision(2)
                        << std::setw(10) << strike
                        << std::setw(12) << maturity
                        << std::setw(12) << (true_vol * 100) << "%"
                        << std::setw(12) << market_price << std::endl;
            }
        }
    }

    // Calibrate the surface
    std::cout << "\nCalibrating volatility surface...\n";
    ImpliedVolSurface surface(
        ImpliedVolSurface::SurfaceType::LOG_MONEYNESS_SPACE,
        ImpliedVolSurface::InterpolationMethod::BICUBIC_SPLINE,
        risk_free_rate
    );

    if (surface.calibrate(market_quotes)) {
        std::cout << "✓ Calibration successful!\n";
        std::cout << "  Calibrated " << surface.get_calibrated_points().size()
                << " volatility points\n";

        // Get calibration statistics
        auto stats = surface.get_calibration_stats(market_quotes);
        std::cout << "\nCalibration Statistics:\n";
        std::cout << "  Mean pricing error:  $" << stats.mean_error << std::endl;
        std::cout << "  Max pricing error:   $" << stats.max_error << std::endl;
        std::cout << "  RMSE:                $" << stats.rmse << std::endl;
        std::cout << "  Points calibrated:   " << stats.num_points << std::endl;
    } else {
        std::cout << "✗ Calibration failed!\n";
    }
}

void example_volatility_interpolation() {
    std::cout << "\n";
    print_separator();
    std::cout << "EXAMPLE 2: Volatility Interpolation" << std::endl;
    print_separator();

    double spot = 100.0;
    double risk_free_rate = 0.03;

    // Create market data with gaps
    std::vector<OptionQuote> quotes;
    std::vector<std::pair<double, double> > strike_maturity_pairs = {
        {90, 0.5}, {100, 0.5}, {110, 0.5},
        {90, 1.0}, {100, 1.0}, {110, 1.0},
        {90, 2.0}, {100, 2.0}, {110, 2.0}
    };

    for (const auto &[strike, maturity]: strike_maturity_pairs) {
        double vol = 0.25; // Flat vol for simplicity
        double price = BlackScholes::call_price(spot, strike, risk_free_rate, vol, maturity);

        OptionQuote quote;
        quote.strike = strike;
        quote.maturity = maturity;
        quote.market_price = price;
        quote.spot = spot;
        quote.forward = spot * std::exp(risk_free_rate * maturity);
        quote.is_call = true;
        quotes.push_back(quote);
    }

    ImpliedVolSurface surface(
        ImpliedVolSurface::SurfaceType::LOG_MONEYNESS_SPACE,
        ImpliedVolSurface::InterpolationMethod::BICUBIC_SPLINE,
        risk_free_rate
    );

    surface.calibrate(quotes);

    std::cout << "\nInterpolating volatility for intermediate strikes and maturities:\n\n";
    std::cout << std::left << std::setw(12) << "Strike"
            << std::setw(12) << "Maturity"
            << std::setw(15) << "Implied Vol" << std::endl;
    std::cout << std::string(39, '-') << std::endl;

    // Test interpolation at various points
    std::vector<std::pair<double, double> > test_points = {
        {95, 0.75}, // Interpolated
        {100, 0.75}, // Interpolated maturity
        {105, 1.5}, // Fully interpolated
        {110, 2.0} // Calibrated point
    };

    for (const auto &[strike, maturity]: test_points) {
        double forward = spot * std::exp(risk_free_rate * maturity);
        try {
            double vol = surface.get_volatility(strike, maturity, forward);
            std::cout << std::fixed << std::setprecision(2)
                    << std::setw(12) << strike
                    << std::setw(12) << maturity
                    << std::setw(15) << (vol * 100) << "%" << std::endl;
        } catch (const std::exception &e) {
            std::cout << std::setw(12) << strike
                    << std::setw(12) << maturity
                    << "Error: " << e.what() << std::endl;
        }
    }
}

void example_comparison_of_methods() {
    std::cout << "\n";
    print_separator();
    std::cout << "EXAMPLE 3: Comparison of Interpolation Methods" << std::endl;
    print_separator();

    double spot = 100.0;
    double risk_free_rate = 0.04;

    // Create sample data
    std::vector<OptionQuote> quotes;
    for (double maturity: {0.5, 1.0, 2.0}) {
        for (double strike: {90, 100, 110}) {
            double vol = 0.22;
            double price = BlackScholes::call_price(spot, strike, risk_free_rate, vol, maturity);

            OptionQuote quote;
            quote.strike = strike;
            quote.maturity = maturity;
            quote.market_price = price;
            quote.spot = spot;
            quote.forward = spot * std::exp(risk_free_rate * maturity);
            quote.is_call = true;
            quotes.push_back(quote);
        }
    }

    // Test different interpolation methods
    std::vector<std::pair<std::string, ImpliedVolSurface::InterpolationMethod> > methods = {
        {"Bilinear", ImpliedVolSurface::InterpolationMethod::BILINEAR},
        {"Bicubic Spline", ImpliedVolSurface::InterpolationMethod::BICUBIC_SPLINE}
    };

    double test_strike = 105;
    double test_maturity = 1.5;
    double test_forward = spot * std::exp(risk_free_rate * test_maturity);

    std::cout << "\nInterpolating volatility at Strike=" << test_strike
            << ", Maturity=" << test_maturity << ":\n\n";
    std::cout << std::left << std::setw(20) << "Method"
            << std::setw(15) << "Implied Vol" << std::endl;
    std::cout << std::string(35, '-') << std::endl;

    for (const auto &[name, method]: methods) {
        ImpliedVolSurface surface(
            ImpliedVolSurface::SurfaceType::LOG_MONEYNESS_SPACE,
            method,
            risk_free_rate
        );

        surface.calibrate(quotes);

        try {
            double vol = surface.get_volatility(test_strike, test_maturity, test_forward);
            std::cout << std::fixed << std::setprecision(4)
                    << std::setw(20) << name
                    << std::setw(15) << (vol * 100) << "%" << std::endl;
        } catch (const std::exception &e) {
            std::cout << std::setw(20) << name
                    << "Error: " << e.what() << std::endl;
        }
    }
}

void example_implied_vol_calculation() {
    std::cout << "\n";
    print_separator();
    std::cout << "EXAMPLE 4: Direct Implied Volatility Calculation" << std::endl;
    print_separator();

    double spot = 100.0;
    double strike = 105.0;
    double risk_free_rate = 0.05;
    double maturity = 1.0;
    double true_vol = 0.25;

    // Generate theoretical option price
    double call_price = BlackScholes::call_price(spot, strike, risk_free_rate, true_vol, maturity);

    std::cout << "\nOption Parameters:\n";
    std::cout << "  Spot:         $" << spot << std::endl;
    std::cout << "  Strike:       $" << strike << std::endl;
    std::cout << "  Risk-free:    " << (risk_free_rate * 100) << "%" << std::endl;
    std::cout << "  Maturity:     " << maturity << " years" << std::endl;
    std::cout << "  True Vol:     " << (true_vol * 100) << "%" << std::endl;
    std::cout << "  Call Price:   $" << call_price << std::endl;

    // Compute implied volatility from the price
    std::cout << "\nRecovering implied volatility from market price...\n";

    try {
        double implied_vol = BlackScholes::implied_volatility(
            call_price, spot, strike, risk_free_rate, maturity, true
        );

        std::cout << "✓ Implied Vol (Newton-Raphson): "
                << (implied_vol * 100) << "%" << std::endl;
        std::cout << "  Error: " << std::abs(implied_vol - true_vol) * 10000 << " bps\n";

        // Also try Brent's method
        double implied_vol_brent = BlackScholes::implied_volatility_brent(
            call_price, spot, strike, risk_free_rate, maturity, true
        );

        std::cout << "✓ Implied Vol (Brent's method):  "
                << (implied_vol_brent * 100) << "%" << std::endl;
        std::cout << "  Error: " << std::abs(implied_vol_brent - true_vol) * 10000 << " bps\n";
    } catch (const std::exception &e) {
        std::cout << "✗ Error: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "\n";
    print_separator();
    std::cout << "         VOLATILITY SURFACE CALIBRATION EXAMPLES" << std::endl;
    print_separator();

    try {
        example_basic_calibration();
        example_volatility_interpolation();
        example_comparison_of_methods();
        example_implied_vol_calculation();

        std::cout << "\n";
        print_separator();
        std::cout << "All examples completed successfully!" << std::endl;
        print_separator();
        std::cout << "\n";
    } catch (const std::exception &e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "VOL_OK" << std::endl;
    return 0;
}

