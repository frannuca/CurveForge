//
// Created by Francisco Nunez on 08.11.2025.
//

#ifndef CURVEFORGE_PILLAR_H
#define CURVEFORGE_PILLAR_H
#include "time/instant.h"
#include "constants.h"

namespace curve {
    class Pillar {
    public:
        Pillar(std::chrono::days &&dt, double value);

        Pillar(const std::chrono::days &dt, double value);

        Pillar(const Pillar &other);

        Pillar(Pillar &&other) noexcept;

        Pillar() = delete;

        // Make the class assignable so vector erase/move-shift works
        Pillar &operator=(Pillar &) = default;

        Pillar &operator=(Pillar &&) noexcept = default;

        [[nodiscard]] Pillar create_new(double new_value) const;

        [[nodiscard]] const double &get_value() const;

        [[nodiscard]] const time::Instant &get_time(const time::Instant &t0) const;

        [[nodiscard]] const std::chrono::days &get_dt() const;

        // Equality and comparison operators
        inline bool operator==(const Pillar &other) const noexcept {
            auto dt = this->time > other.time ? this->time - other.time : other.time - this->time;
            return dt < EPS_INSTANT_MS && std::abs(value - other.value) < EPS_RATE;
        }

        inline bool operator!=(const Pillar &other) const noexcept {
            return !(*this == other);
        }

        // Strict weak ordering: compare by time first, then by value
        inline bool operator<(const Pillar &other) const noexcept {
            return this->time < other.time;
        }

        inline bool operator>(const Pillar &other) const noexcept { return other < *this; }
        inline bool operator<=(const Pillar &other) const noexcept { return !(other < *this); }
        inline bool operator>=(const Pillar &other) const noexcept { return !(*this < other); }

    private:
        std::chrono::days time;
        double value;
    };
}
#endif //CURVEFORGE_PILLAR_H
