
#include <utility>
#include "time/instant.h"
#include "curve/Pillar.h"

namespace curve {
    Pillar::Pillar(const time::Date &t, double value)
        : date(t), value(value) {
    }


    Pillar Pillar::create_new(double new_value) const {
        return {this->date, new_value};
    }

    const double &Pillar::get_value() const {
        return value;
    }

    const time::Date &Pillar::get_time() const {
        return date;
    }
} // namespace curve
