#ifndef LOSS_H
#define LOSS_H

// Forward declaration of Matrix struct
typedef struct Matrix Matrix;

// Regression loss functions
double mean_squared_error(Matrix* y, Matrix* y_pred);
double mean_absolute_error(Matrix* y, Matrix* y_pred);

// Classification loss functions (currently only BCE)
double binary_cross_entropy(Matrix* y, Matrix* y_pred);

#endif