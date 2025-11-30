//
// Created by Francisco Nunez on 30.11.2025.
//

#ifndef CURVEFORGE_OPTIONQUOTE_H
#define CURVEFORGE_OPTIONQUOTE_H

namespace curve::volatility {
    /**
     * @brief Structure to hold option market data for calibration
     */
    struct OptionQuote {
        double strike; // Strike price
        double maturity; // Time to maturity in years
        double market_price; // Market price of the option
        double spot; // Spot price at quote time
        double forward; // Forward price (if available)
        bool is_call; // True for call, false for put
        double moneyness; // K/F or ln(K/F)

        OptionQuote() : strike(0), maturity(0), market_price(0),
                        spot(0), forward(0), is_call(true), moneyness(0) {
        }
    };
}

#endif //CURVEFORGE_OPTIONQUOTE_H
