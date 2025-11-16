//
// Created by Francisco Nunez on 08.11.2025.
//

#ifndef CURVEFORGE_PILLAR_H
#define CURVEFORGE_PILLAR_H
#include "time/instant.h"
#include "constants.h"
#include "time/date.hpp"

namespace curve {
    class Pillar {
    public:
        Pillar(const time::Date &t, double value);

        Pillar(const Pillar &other) = default;

        Pillar(Pillar &&other) noexcept = default;

        Pillar() = delete;

        // Make the class assignable so vector erase/move-shift works
        Pillar &operator=(const Pillar &) = default;

        Pillar &operator=(Pillar &&) noexcept = default;

        [[nodiscard]] Pillar create_new(double new_value) const;

        [[nodiscard]] const double &get_value() const;

        [[nodiscard]] const time::Date &get_time() const;

        // Equality and comparison operators
        inline bool operator==(const Pillar &other) const noexcept {
            auto dt = this->date > other.date
                          ? std::chrono::sys_days(this->date) - std::chrono::sys_days(other.date)
                          : std::chrono::sys_days(other.date) - std::chrono::sys_days(this->date);
            return dt < EPS_INSTANT && std::abs(value - other.value) < EPS_RATE;
        }

        inline bool operator!=(const Pillar &other) const noexcept {
            return !(*this == other);
        }

        // Strict weak ordering: compare by time first, then by value
        inline bool operator<(const Pillar &other) const noexcept {
            return this->date < other.date;
        }

        inline bool operator>(const Pillar &other) const noexcept { return other < *this; }
        inline bool operator<=(const Pillar &other) const noexcept { return !(other < *this); }
        inline bool operator>=(const Pillar &other) const noexcept { return !(*this < other); }

    private:
        time::Date date;
        double value;
    };
}
#endif //CURVEFORGE_PILLAR_H
