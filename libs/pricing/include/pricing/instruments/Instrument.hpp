#pragma once
#include <string>
#include <memory>

namespace pricing {
    enum class InstrumentType {
        Bond,
        Option,
        BarrierOption,
        Swap,
        Future,
        Cds,
        xSwap,
        Fra
        // Add more instrument types as needed
    };

    inline std::string to_string(InstrumentType type) {
        switch (type) {
            case InstrumentType::Bond: return "Bond";
            case InstrumentType::Option: return "Option";
            case InstrumentType::Swap: return "Swap";
            case InstrumentType::Future: return "Future";
            default: return "Unknown";
        }
    }

    class Instrument {
    public:
        virtual ~Instrument() = default;

        virtual double maturity() const = 0; // in years
        virtual double notional() const = 0;

        virtual InstrumentType type() const = 0;
    };

    using InstrumentPtr = std::shared_ptr<Instrument>;
} // namespace pricing
