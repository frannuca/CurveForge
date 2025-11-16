//
// Created by Francisco Nunez on 18.10.2025.
//

#include "instruments/Instrument.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <utility>

namespace curve::instruments {
    Instrument::Instrument(std::string ccy) : currency_(std::move(ccy)) {
        id_ = to_string(boost::uuids::random_generator{}());
    }
} // namespace instruments
