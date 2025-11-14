
#include <utility>
#include "time/instant.h"
#include "curve/Pillar.h"

namespace curve {
    Pillar::Pillar(std::chrono::days &&dt, double value)
        : time(std::move(dt)), value(value) {
    }

    Pillar::Pillar(const std::chrono::days &dt, double value)
        : time(dt), value(value) {
    }

    Pillar::Pillar(const Pillar &other)
        : time(other.time), value(other.value) {
    }

    Pillar::Pillar(Pillar &&other) noexcept
        : time(std::move(other.time)), value(std::move(other.value)) {
    }

    Pillar Pillar::create_new(double new_value) const {
        // Provide a thread-local copy with the updated value to avoid returning a dangling reference.
        return Pillar(this->time, new_value);
    }

    const double &Pillar::get_value() const {
        return value;
    }

    const std::chrono::days &Pillar::get_dt() const {
        return time;
    }

    const time::Instant &Pillar::get_time(const time::Instant &t0) const {
        return (t0 + time);
    }
} // namespace curve
