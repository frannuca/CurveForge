
#include <utility>
#include "time/instant.h"
#include "curve/Pillar.h"

namespace curve {
    Pillar::Pillar(time::Instant &&time_to_maturity, double value)
        : time(std::move(time_to_maturity)), value(value) {
    }

    Pillar::Pillar(const time::Instant &time_to_maturity, double value)
        : time(time_to_maturity), value(value) {
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

    const time::Instant &Pillar::get_time() const {
        return time;
    }
} // namespace curve
