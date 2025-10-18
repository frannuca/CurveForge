//
// Created by Francisco Nunez on 18.10.2025.
//

#include "../include/instruments/InstrumentBase.h"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace curve::instruments {
    InstrumentBase::InstrumentBase(time::Date maturity,
                                   time::Date start_date) : maturity_date(maturity), start_date(start_date) {
        // Generate a random (v4) UUID for the instrument id
        boost::uuids::uuid u = boost::uuids::random_generator()();
        id_ = boost::uuids::to_string(u);
    }
} // namespace instruments
