#include "matrix.h"

#ifndef ACTIVATION_H
#define ACTIVATION_H

// Applies a given activation function to each element in a matrix.
void apply_activation(Matrix* matrix, double (*activation)(double));

// 1 / (1 + e^{-x})
double sigmoid(double x);

// (e^{x} - e^{-x}) / (e^{x} + e^{-x})
double tanh(double x);

// max(0, x)
double ReLu(double x);

#endif