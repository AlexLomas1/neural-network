#ifndef LOSS_H
#define LOSS_H

// Forward declaration of Matrix struct
typedef struct Matrix Matrix;

typedef struct LossFunc {
    double (*func_ptr)(Matrix*, Matrix*);
    Matrix* (*derivative_ptr)(Matrix*, Matrix*);
} LossFunc;

extern const LossFunc MSE;
extern const LossFunc MAE;
extern const LossFunc BCE;

// Regression loss functions
double mean_squared_error(Matrix* y, Matrix* y_pred);
Matrix* mean_squared_error_derivative(Matrix* y, Matrix* y_pred);

double mean_absolute_error(Matrix* y, Matrix* y_pred);
Matrix* mean_absolute_error_derivative(Matrix* y, Matrix* y_pred);

// Classification loss functions (currently only BCE)
double binary_cross_entropy(Matrix* y, Matrix* y_pred);
Matrix* binary_cross_entropy_derivative(Matrix* y, Matrix* y_pred);

#endif