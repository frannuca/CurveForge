#pragma once

#include <string>
#include "time/instant.h"
#include "time/date.hpp"
#include <optional>

namespace curve::instruments {
    class Instrument {
    public:
        Instrument(std::string &&ccy) : currency_(std::move(ccy)) {
        }

        Instrument(const std::string &ccy) : currency_(ccy) {
        }

        virtual std::string name() const = 0;

        // Read-only property: identifier
        const std::string &id() const { return id_; }

        const std::string currency() const;

        virtual ~Instrument() {
        }

    protected:
        // Allow derived classes to set the id during construction or initialization
        void set_id(std::string id) { id_ = std::move(id); }

    private:
        std::string id_;
        std::string currency_;
    };
} // namespace instruments
