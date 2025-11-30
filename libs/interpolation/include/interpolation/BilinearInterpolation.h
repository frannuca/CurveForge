//
// Created by Francisco Nunez on 30.11.2025.
//

#ifndef CURVEFORGE_BILINEAR_H
#define CURVEFORGE_BILINEAR_H
#include <vector>
#include <Eigen/Dense>

namespace interpolation {
    class BilinearInterpolation {
    public:
        BilinearInterpolation(std::vector<double> &&x, std::vector<double> &&y, Eigen::MatrixXd &&z);

        BilinearInterpolation(const std::vector<double> &x, const std::vector<double> &y, const Eigen::MatrixXd &z);

        double interpolate(const double &x, const double &y) const;

    private:
        std::vector<double> x_, y_;
        Eigen::MatrixXd z_;
    };
}


#endif //CURVEFORGE_BILINEAR_H
