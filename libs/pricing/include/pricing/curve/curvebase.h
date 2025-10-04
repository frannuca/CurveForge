//
// Created by Francisco Nunez on 04.10.2025.
//

#ifndef CURVEFORGE_CURVEBASE_H
#define CURVEFORGE_CURVEBASE_H

namespace pricing {
    class curvebase {
        virtual double discount(double t) const = 0;

        virtual double zero(double t) const = 0;

        virtual double get_forward(double t, double dT) const;

        virtual double instantaneous_forward(double t) const;
    };
}

#endif //CURVEFORGE_CURVEBASE_H
