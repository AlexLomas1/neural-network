#ifndef ACTIVATION_H
#define ACTIVATION_H

// 1 / (1 + e^{-x})
double sigmoid(double x);

// (e^{x} - e^{-x}) / (e^{x} + e^{-x})
// Named custom to prevent conflict with math.h tanh function.
double tanh_custom(double x);

// max(0, x)
double ReLu(double x);

#endif