#include <math.h>
#include "activation.h"

double sigmoid(double x) {
    return (1 / (1 + exp(-x)));
}

double tanh_custom(double x) {
    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
}

double ReLu(double x) {
    if (x > 0) {
        return x;
    }
    return 0;
}