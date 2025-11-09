//
// Created by Francisco Nunez on 28.09.2025.
//

#ifndef CURVEFORGE_INSTANT_H
#define CURVEFORGE_INSTANT_H
#include <chrono>
#include <cstdint>

namespace curve::time {
    using Instant = std::chrono::system_clock::time_point;

    // Return signed milliseconds difference (a - b) in ms. May be negative if a < b.
    inline std::int64_t milliseconds_between(const Instant &a, const Instant &b) noexcept {
        return std::chrono::duration_cast<std::chrono::milliseconds>(a - b).count();
    }
}
#endif //CURVEFORGE_INSTANT_H
