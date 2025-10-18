#pragma once

#include <string>
#include "time/instant.h"
#include "time/date.hpp"
#include <optional>

namespace curve::instruments {
    class InstrumentBase {
    public:
        InstrumentBase() = delete;

        InstrumentBase(time::Date maturity, time::Date start_date);

        virtual ~InstrumentBase() = default;

        // Basic interface
        virtual double notional() const = 0;

        // Present value - pure virtual so derived classes can override
        virtual double pv() const = 0;

        virtual std::string name() const = 0;

        // Read-only property: identifier
        const std::string &id() const { return id_; }
        const time::Date maturity_date;
        const time::Date start_date;

    protected:
        // Allow derived classes to set the id during construction or initialization
        void set_id(std::string id) { id_ = std::move(id); }

    private:
        std::string id_;
    };
} // namespace instruments
