//
// Created by Francisco Nunez on 01.11.2025.
//

#ifndef CURVEFORGE_LINEAR_H
#define CURVEFORGE_LINEAR_H
#include <vector>


class LinearInterpolation {
public:
   static double lininterp(const std::vector<double> &x, const std::vector<double> &y, double xq);
};


#endif //CURVEFORGE_LINEAR_H
