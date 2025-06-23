#include <stdlib.h>
#include <math.h>
#include "maths/loss.h"
#include "maths/matrix.h"

const LossFunc MSE = {&mean_squared_error, &mean_squared_error_derivative};
const LossFunc MAE = {&mean_absolute_error, &mean_absolute_error_derivative};
const LossFunc BCE = {&binary_cross_entropy, &binary_cross_entropy_derivative};

double mean_squared_error(const Matrix* y, const Matrix* y_pred) {
    double squared_diff_sum = 0.0;
    for (int col_count=0; col_count < y->cols; col_count++) { // Each column represents a sample
        for (int row_count=0; row_count < y->rows; row_count++) {
            double diff = get_element(y, row_count, col_count) - get_element(y_pred, row_count, col_count);
            squared_diff_sum += (diff * diff);
        }
    }

    return (squared_diff_sum / (y->rows * y->cols));
}

Matrix mean_squared_error_derivative(const Matrix* y, const Matrix* y_pred) {
    // Creates matrix of MSE's partial derivatives with respect to each of the predictions.
    Matrix gradient_matrix = create_matrix(y->rows, y->cols);
    
    for (int col_count=0; col_count < y->cols; col_count++) { // Each column represents a sample
        for (int row_count=0; row_count < y->rows; row_count++) {
            double y_i = get_element(y, row_count, col_count);
            double y_pred_i = get_element(y_pred, row_count, col_count);

            double grad = (2.0/(y->rows * y->cols)) * (y_pred_i - y_i);
            set_element(&gradient_matrix, row_count, col_count, grad);
        }
    }

    return gradient_matrix;
}

double mean_absolute_error(const Matrix* y, const Matrix* y_pred) {
    double abs_diff_sum = 0.0;
    for (int col_count=0; col_count < y->cols; col_count++) { // Each column represents a sample
        for (int row_count=0; row_count < y->rows; row_count++) {
            double diff = get_element(y, row_count, col_count) - get_element(y_pred, row_count, col_count);
            abs_diff_sum += fabs(diff);
        }
    }

    return (abs_diff_sum / (y->rows * y->cols));
}

Matrix mean_absolute_error_derivative(const Matrix* y, const Matrix* y_pred) {
    // Creates matrix of MAE's partial derivatives with respect to each of the predictions.
    Matrix gradient_matrix = create_matrix(y->rows, y->cols);

    for (int col_count=0; col_count < y->cols; col_count++) { // Each column represents a sample
        for (int row_count=0; row_count < y->rows; row_count++) {
            double y_i = get_element(y, row_count, col_count);
            double y_pred_i = get_element(y_pred, row_count, col_count);

            if (y_i == y_pred_i) {
                // MAE's derivative is actually undefined at y_i = y_pred_i, but here it is considered to be 0.
                // Setting to zero is unneeded as matrix initialised to all zeros, but want to make it explicit.
                set_element(&gradient_matrix, row_count, col_count, 0); 
            }
            else {
                double grad = (y_i > y_pred_i) ? -1.0 / (y->rows * y->cols) : 1.0 / (y->rows * y->cols);
                set_element(&gradient_matrix, row_count, col_count, grad);
            }
        }
    }

    return gradient_matrix;
}

double binary_cross_entropy(const Matrix* y, const Matrix* y_pred) {
    const double epsilon = 1e-15;
    double sum = 0.0;
    
    for (int col_count=0; col_count < y->cols; col_count++) { // Each column represents a sample
        for (int row_count=0; row_count < y->rows; row_count++) {
            double y_i = get_element(y, row_count, col_count);
            double y_pred_i = get_element(y_pred, row_count, col_count);

            // Clipping values of y_i_pred into range [epsilon, 1.0-epsilon], to prevent calculating log(0)
            if (y_pred_i < epsilon) {
                y_pred_i = epsilon;
            }
            else if (y_pred_i > (1.0-epsilon)) {
                y_pred_i = 1.0 - epsilon;
            }

            // Note that log function from math.h has base e
            sum += -(y_i * log(y_pred_i) + (1.0-y_i)*log(1.0-y_pred_i));
        }
    }

    return (sum / (y->rows * y->cols));
}

Matrix binary_cross_entropy_derivative(const Matrix* y, const Matrix* y_pred) {
    // Creates matrix of BCE's partial derivatives with respect to each of the predictions.
    const double epsilon = 1e-15;
    Matrix gradient_matrix = create_matrix(y->rows, y->cols);

    for (int col_count=0; col_count < y->cols; col_count++) {
        for (int row_count=0; row_count < y->rows; row_count++) {
            double y_i = get_element(y, row_count, col_count);
            double y_pred_i = get_element(y_pred, row_count, col_count);
        
            if (y_pred_i < epsilon) {
                y_pred_i = epsilon;
            }
            else if (y_pred_i > 1.0 - epsilon) {
                y_pred_i = 1.0 - epsilon;
            }

            double grad = (-1.0/(y->rows * y->cols)) * ((y_i/y_pred_i) - ((1-y_i) / (1-y_pred_i)));
            set_element(&gradient_matrix, row_count, col_count, grad);
        }
    }

    return gradient_matrix;
}