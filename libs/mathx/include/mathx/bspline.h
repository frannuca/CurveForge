//
// Created by Francisco Nunez on 20.09.2025.
//

#ifndef CURVEFORGE_BSPLINE_H
#define CURVEFORGE_BSPLINE_H
#include <vector>
#include <stdexcept>
#include <Eigen/Dense>

namespace mathx::interpolation {
    class bspline {
    public :
        bspline(const std::vector<Eigen::VectorXd>& control_points, const size_t degree);
        bspline(const std::vector<Eigen::VectorXd>& control_points, size_t degree, const std::vector<double>& knots); // explicit knots
        bspline() = delete;

        bspline(const bspline &other);

        bspline &operator=(const bspline &other);

        bspline(bspline &&other) noexcept;

        bspline &operator=(bspline &&other) noexcept;
        ~bspline(){}

        std::vector<double> basis_function(double u) const;
        size_t find_span(double u) const;
        Eigen::VectorXd evaluate(double u) const;

        // Interpolate given data points (they will be passed exactly). Parameterization: "uniform" or "chord".
        static bspline interpolate(const std::vector<Eigen::VectorXd>& data_points,
                                   size_t degree,
                                   const std::string& parameterization = "chord");

        // Smoothing (penalized) interpolation: lambda=0 -> exact, lambda>0 applies second-difference penalty.
        static bspline smooth_interpolate(const std::vector<Eigen::VectorXd>& data_points,
                                          size_t degree,
                                          double lambda,
                                          const std::string& parameterization = "chord");
    private:
        std::vector<double> knots_;
        std::vector<Eigen::VectorXd> control_points_;
        size_t n_;
        size_t p_;

        // Clamped open-uniform knot vector on [0,1]
        // cpCount control points, degree p => size(U) = (cpCount -1) + p + 2 = cpCount + p +1
        std::vector<double> clamped_knots(size_t cpCount, size_t degree);

    };
}
#endif //CURVEFORGE_BSPLINE_H
