#pragma once

#include <string>
#include "time/instant.h"
#include "time/date.hpp"
#include <optional>

namespace curve::instruments {
    class Instrument {
    public:
        explicit Instrument(std::string ccy);

        [[nodiscard]] virtual std::string name() const = 0;

        // Read-only property: identifier
        [[nodiscard]] const std::string &id() const { return id_; }

        [[nodiscard]] std::string currency() const;

        virtual ~Instrument() = default;

    protected:
        // Allow derived classes to set the id during construction or initialization
        void set_id(std::string id) { id_ = std::move(id); }

    private:
        std::string id_;
        std::string currency_;
    };

    inline std::string Instrument::currency() const {
        return currency_;
    }
} // namespace instruments
