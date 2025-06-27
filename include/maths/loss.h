#ifndef LOSS_H
#define LOSS_H

#include "maths/matrix.h" // For Matrix struct and matrix operations

typedef struct LossFunc {
    double (*func_ptr)(const Matrix*, const Matrix*);
    Matrix (*derivative_ptr)(const Matrix*, const Matrix*);
} LossFunc;

extern const LossFunc MSE;
extern const LossFunc MAE;
extern const LossFunc BCE;
extern const LossFunc CCE;

// Regression loss functions
double mean_squared_error(const Matrix* y, const Matrix* y_pred);
Matrix mean_squared_error_derivative(const Matrix* y, const Matrix* y_pred);

double mean_absolute_error(const Matrix* y, const Matrix* y_pred);
Matrix mean_absolute_error_derivative(const Matrix* y, const Matrix* y_pred);

// Classification loss functions
double binary_cross_entropy(const Matrix* y, const Matrix* y_pred);
Matrix binary_cross_entropy_derivative(const Matrix* y, const Matrix* y_pred);

double categorical_cross_entropy(const Matrix* y, const Matrix* y_pred);
Matrix categorical_cross_entropy_derivative(const Matrix* y, const Matrix* y_pred);

#endif