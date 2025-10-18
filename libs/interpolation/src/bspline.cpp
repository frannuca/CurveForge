//
// Created by Francisco Nunez on 20.09.2025.
//

#include "interpolation/bspline.h"
using namespace interpolation;
#include <string>

bspline::bspline(const std::vector<Eigen::VectorXd> &control_points, const size_t degree) : n_(control_points.size()),
    p_(degree), control_points_(control_points) {
    if (degree == 0) throw std::invalid_argument("degree must be > 0");
    if (n_ < degree + 1) throw std::invalid_argument("insufficient control points for degree");
    this->knots_ = clamped_knots(n_, degree);
}

bspline::bspline(const bspline &other) : n_(other.n_), p_(other.p_), control_points_(other.control_points_),
                                         knots_(other.knots_) {
}

bspline &bspline::operator=(const bspline &other) {
    n_ = other.n_;
    p_ = other.p_;
    knots_ = other.knots_;
    control_points_ = other.control_points_;
    return *this;
}

bspline::bspline(bspline &&other) noexcept : n_(other.n_), p_(other.p_) {
    knots_ = std::move(other.knots_);
    control_points_ = std::move(other.control_points_);
}

bspline &bspline::operator=(bspline &&other) noexcept {
    knots_ = std::move(other.knots_);
    control_points_ = std::move(other.control_points_);
    return *this;
}

size_t bspline::find_span(double u) const {
    // Open uniform clamped: knots_[0..p_] = 0, knots_[m-p_..m] = 1
    size_t cpCount = n_; // number of control points
    size_t nLast = cpCount - 1; // highest control point index
    if (u >= 1.0) return nLast; // convention: last span maps to last control interval
    if (u <= 0.0) return p_; // first non-zero span index

    size_t low = p_;
    size_t high = nLast;
    size_t mid = (low + high) / 2;
    int iter = 0;
    const int maxiter = 100;
    while (!(u >= knots_[mid] && u < knots_[mid + 1]) && ++iter < maxiter) {
        if (u < knots_[mid]) {
            if (mid == 0) break; // safety
            high = mid - 1;
        } else {
            low = mid + 1;
        }
        mid = (low + high) / 2;
    }
    if (iter == maxiter) throw std::runtime_error("find_span max iterations");
    return mid;
}

Eigen::VectorXd bspline::evaluate(double u) const {
    // Clamp u into [0,1]
    if (u < 0.0) u = 0.0;
    if (u > 1.0) u = 1.0;
    int p = static_cast<int>(p_);
    const auto &U = knots_;
    int k = static_cast<int>(find_span(u));
    const auto &P = control_points_;
    // working copy of the p+1 relevant control points
    std::vector<Eigen::VectorXd> d(p + 1);
    for (int j = 0; j <= p; ++j) d[j] = P[k - p + j];

    for (int r = 1; r <= p; ++r) {
        for (int j = p; j >= r; --j) {
            double num = u - U[k - p + j];
            double den = U[k + 1 + j - r] - U[k - p + j];
            double alpha = (den == 0.0) ? 0.0 : (num / den);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }
    return d[p];
}

std::vector<double> bspline::clamped_knots(size_t cpCount, size_t degree) {
    // Open uniform clamped: size = cpCount + degree + 1
    size_t m = cpCount + degree; // last index
    std::vector<double> U(m + 1);
    for (size_t i = 0; i <= m; ++i) {
        if (i <= degree) U[i] = 0.0;
        else if (i >= cpCount) U[i] = 1.0;
        else U[i] = (static_cast<double>(i - degree)) / (cpCount - degree);
    }
    return U;
}

std::vector<double> bspline::basis_function(double u) const {
    auto k = find_span(u);
    int p = static_cast<int>(p_);
    const auto &U = knots_;
    std::vector<double> N(p + 1, 0.0);
    std::vector<double> left(p + 1), right(p + 1);
    N[0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j] = u - U[k + 1 - j];
        right[j] = U[k + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            double den = right[r + 1] + left[j - r];
            double temp = (den == 0.0) ? 0.0 : N[r] / den;
            N[r] = saved + temp * right[r + 1];
            saved = temp * left[j - r];
        }
        N[j] = saved;
    }
    return N;
}

bspline::bspline(const std::vector<Eigen::VectorXd> &control_points, size_t degree, const std::vector<double> &knots)
    : n_(control_points.size()), p_(degree), control_points_(control_points), knots_(knots) {
    if (degree == 0) throw std::invalid_argument("degree must be > 0");
    if (n_ < degree + 1) throw std::invalid_argument("insufficient control points for degree");
    size_t expected = n_ + degree + 1; // size of knots
    if (knots_.size() != expected) throw std::invalid_argument("knot vector size mismatch");
    if (knots_.front() != 0.0 || knots_.back() != 1.0)
        throw std::invalid_argument(
            "knot vector must be clamped to [0,1]");
}

// Helper: parameterization
static std::vector<double> parameterize(const std::vector<Eigen::VectorXd> &pts, const std::string &method) {
    size_t m = pts.size();
    std::vector<double> u(m, 0.0);
    if (m == 0) return u;
    if (method == "uniform") {
        for (size_t i = 0; i < m; ++i) u[i] = (m == 1) ? 0.0 : double(i) / double(m - 1);
        return u;
    }
    // chord length (default)
    double total = 0.0;
    for (size_t i = 1; i < m; ++i) total += (pts[i] - pts[i - 1]).norm();
    if (total == 0.0) {
        for (size_t i = 0; i < m; ++i) u[i] = (m == 1) ? 0.0 : double(i) / double(m - 1);
        return u;
    }
    double acc = 0.0;
    for (size_t i = 1; i < m - 1; ++i) {
        acc += (pts[i] - pts[i - 1]).norm();
        u[i] = acc / total;
    }
    u.back() = 1.0;
    u.front() = 0.0;
    return u;
}

// Helper: interpolation knot vector per The NURBS Book Algorithm A9.1
static std::vector<double> interpolation_knots(const std::vector<double> &u, size_t degree) {
    size_t m = u.size(); // also number of control points
    size_t p = degree;
    size_t knotSize = m + p + 1; // n = m-1, size = n + p + 2 = (m-1)+p+2 = m+p+1
    std::vector<double> U(knotSize);
    for (size_t i = 0; i <= p; ++i) U[i] = 0.0;
    for (size_t i = 0; i <= p; ++i) U[knotSize - 1 - i] = 1.0;
    if (m > p + 1) {
        for (size_t j = 1; j < m - p; ++j) {
            // j = 1 .. m-p-1 inclusive
            double s = 0.0;
            for (size_t i = j; i < j + p; ++i) s += u[i];
            U[j + p] = s / double(p);
        }
    }
    return U;
}

std::unique_ptr<bspline> bspline::interpolate(const std::vector<Eigen::VectorXd> &data_points,
                                              size_t degree,
                                              const std::string &parameterization) {
    if (data_points.empty()) throw std::invalid_argument("empty data_points");
    if (degree == 0) throw std::invalid_argument("degree must be >0");
    size_t m = data_points.size();
    if (m < degree + 1) throw std::invalid_argument("need at least degree+1 data points");

    auto u = parameterize(data_points, parameterization);
    auto knots = interpolation_knots(u, degree);

    // Assemble interpolation matrix (m x m)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, m);
    // For each parameter u_i build row using span and basis values
    // We need a temporary bspline object to leverage existing basis evaluation; build with dummy cps & knots.
    // Control points placeholder (m) to satisfy constructor.
    std::vector<Eigen::VectorXd> dummyCPs;
    dummyCPs.reserve(m);
    for (size_t i = 0; i < m; ++i) dummyCPs.push_back(data_points[0]); // dimension match
    bspline helper(dummyCPs, degree, knots);

    for (size_t i = 0; i < m; ++i) {
        double ui = u[i];
        size_t span = helper.find_span(ui);
        auto N = helper.basis_function(ui); // size p+1, corresponds to span-p .. span
        size_t firstCol = span - degree;
        for (size_t r = 0; r < N.size(); ++r) {
            A(i, firstCol + r) = N[r];
        }
    }

    // RHS matrix (m x dim)
    size_t dim = data_points[0].size();
    Eigen::MatrixXd B(m, dim);
    for (size_t i = 0; i < m; ++i) B.row(i) = data_points[i].transpose();

    Eigen::MatrixXd P = A.fullPivLu().solve(B);
    if ((A * P - B).norm() > 1e-8) throw std::runtime_error("Interpolation solve failed (residual too large)");

    std::vector<Eigen::VectorXd> cps(m);
    for (size_t i = 0; i < m; ++i) cps[i] = P.row(i).transpose();

    return std::make_unique<bspline>(cps, degree, knots);
}

std::unique_ptr<bspline> bspline::smooth_interpolate(const std::vector<Eigen::VectorXd> &data_points,
                                                     size_t degree,
                                                     double lambda,
                                                     const std::string &parameterization) {
    if (lambda <= 0.0) return interpolate(data_points, degree, parameterization);
    if (data_points.empty()) throw std::invalid_argument("empty data_points");
    if (degree == 0) throw std::invalid_argument("degree must be >0");
    size_t m = data_points.size();
    if (m < degree + 1) throw std::invalid_argument("need at least degree+1 data points");

    auto u = parameterize(data_points, parameterization);

    // Decide number of control points (reduce for stronger smoothing)
    // Heuristic: half the data points (at least degree+1, at most m)
    size_t cpCount = std::max<size_t>(degree + 1, std::min(m, (m + degree) / 2));

    // Build interpolation (fitting) knot vector for cpCount control points
    // For smoothing we still use clamped open uniform knots (simpler than averaging interior parameters)
    auto build_knots = [](size_t cpCount_, size_t deg) {
        size_t mlast = cpCount_ + deg; // last index
        std::vector<double> U(mlast + 1);
        for (size_t i = 0; i <= mlast; ++i) {
            if (i <= deg) U[i] = 0.0;
            else if (i >= cpCount_) U[i] = 1.0;
            else U[i] = double(i - deg) / double(cpCount_ - deg);
        }
        return U;
    };
    auto knots = build_knots(cpCount, degree);

    // Assemble basis matrix A (m x cpCount)
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, cpCount);
    std::vector<Eigen::VectorXd> dummyCPs(cpCount, data_points[0]);
    bspline helper(dummyCPs, degree, knots);
    for (size_t i = 0; i < m; ++i) {
        double ui = u[i];
        size_t span = helper.find_span(ui);
        auto N = helper.basis_function(ui);
        size_t firstCol = span - degree;
        for (size_t r = 0; r < N.size(); ++r) {
            size_t col = firstCol + r;
            if (col < cpCount) A(i, col) = N[r];
        }
    }

    size_t dim = data_points[0].size();
    Eigen::MatrixXd B(m, dim);
    for (size_t i = 0; i < m; ++i) B.row(i) = data_points[i].transpose();

    // Second-difference penalty matrix R (cpCount x cpCount)
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(cpCount, cpCount);
    if (cpCount > 3) {
        Eigen::MatrixXd D2 = Eigen::MatrixXd::Zero(cpCount - 2, cpCount);
        for (int i = 0; i < static_cast<int>(cpCount) - 2; ++i) {
            D2(i, i) = 1.0;
            D2(i, i + 1) = -2.0;
            D2(i, i + 2) = 1.0;
        }
        R = D2.transpose() * D2;
    }

    Eigen::MatrixXd ATA = A.transpose() * A;
    Eigen::MatrixXd M = ATA + lambda * R;
    Eigen::MatrixXd RHS = A.transpose() * B;

    Eigen::MatrixXd P = M.ldlt().solve(RHS);

    std::vector<Eigen::VectorXd> cps(cpCount);
    for (size_t i = 0; i < cpCount; ++i) cps[i] = P.row(i).transpose();

    return std::make_unique<bspline>(cps, degree, knots);
}
