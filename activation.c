#include <math.h>
#include "activation.h"

// Custom included in tanh name to prevent conflict with tanh function in math.h
const ActivationFunc sigmoid = {&sigmoid_func, &sigmoid_derivative};
const ActivationFunc tanh_custom = {&tanh_func, &tanh_derivative};
const ActivationFunc ReLu = {&ReLu_func, &ReLu_derivative};

double sigmoid_func(double x) {
    return (1.0 / (1.0 + exp(-x)));
}

double sigmoid_derivative(double x) {
    return (sigmoid_func(x) * (1.0 - sigmoid_func(x)));
}

double tanh_func(double x) {
    // For large magnitudes of x, return the value tanh(x) tends to for the sign of x to prevent overflow
    // with large exponents.
    if (x > 10.0) {
        return 1.0;
    }
    if (x < -10.0) {
        return -1.0;
    }

    return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
}

double tanh_derivative(double x) {
    return (1.0 - (tanh_func(x) * tanh_func(x)));
}

double ReLu_func(double x) {
    if (x > 0.0) {
        return x;
    }
    return 0;
}

double ReLu_derivative(double x) {
    // Note that ReLu's derivative is technically undefined at x = 0, but is taken as being 0 here.
    if (x > 0.0) {
        return 1.0;
    }
    return 0.0;
}