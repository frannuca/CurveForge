//
// Created by Francisco Nunez on 28.09.2025.
//

#ifndef CURVEFORGE_INSTANT_H
#define CURVEFORGE_INSTANT_H
#include <chrono>

namespace curve::time {
        using Instant = std::chrono::system_clock::time_point;
}
#endif //CURVEFORGE_INSTANT_H
