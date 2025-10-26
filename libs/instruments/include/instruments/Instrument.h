#pragma once

#include <string>
#include "time/instant.h"
#include "time/date.hpp"
#include <optional>

namespace curve::instruments {
    class Instrument {
    public:
        virtual std::string name() const = 0;

        // Read-only property: identifier
        const std::string &id() const { return id_; }

        virtual ~Instrument() {
        }

    protected:
        // Allow derived classes to set the id during construction or initialization
        void set_id(std::string id) { id_ = std::move(id); }

    private:
        std::string id_;
    };
} // namespace instruments
