//
// Created by Francisco Nunez on 22.11.2025.
//

#include "curve/FlatRateCurve.h"

curve::FlatRateCurve::FlatRateCurve(time::Date cob_date, double constant_rate)
    : curve::ICurve(
          cob_date,
          {
              {
                  {cob_date, constant_rate},
                  {cob_date + std::chrono::years(100), constant_rate},
              },
          },
          curve::time::create_daycount_convention(curve::time::DayCountConvention::ACT_365F)
      )
      ,
      constant_rate_(constant_rate) {
}

std::string curve::FlatRateCurve::name() const { return "FlatRateCurve"; }
