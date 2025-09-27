#pragma once
#include <vector>

namespace pricing {

struct CurveNode { double t; double df; };

class YieldCurve {
    std::vector<CurveNode> nodes_;
public:
    YieldCurve();
    void add(double t, double df);
    double discount(double t) const;
    const std::vector<CurveNode>& nodes() const { return nodes_; }
};

} // namespace pricing
