#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "maths/activation.h"
#include "maths/matrix.h"

// Softmax is only activation function in the project that isn't element-wise, so it is seperated.
extern const ActivationFunc softmax; 

Matrix softmax_func(const Matrix* x);

Matrix softmax_derivative(const Matrix* x, const Matrix* loss_deriv);

#endif