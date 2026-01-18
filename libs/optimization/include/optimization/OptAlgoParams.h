//
// Created by Francisco Nunez on 18.01.2026.
//

#ifndef CURVEFORGE_OPTALGOPARAMS_H
#define CURVEFORGE_OPTALGOPARAMS_H


struct OptAlgoParams {
    const double ftol;
    const double xtol;
    const int maxeval;

    OptAlgoParams(double ftol_, double xtol_, int maxeval_)
        : ftol(ftol_), xtol(xtol_), maxeval(maxeval_) {
    }
};


#endif //CURVEFORGE_OPTALGOPARAMS_H
