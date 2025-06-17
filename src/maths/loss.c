#include <stdlib.h>
#include <math.h>
#include "maths/loss.h"
#include "maths/matrix.h"

double mean_squared_error(Matrix* y, Matrix* y_pred) {
    double squared_diff_sum = 0;
    for (int row_count=0; row_count < y->rows; row_count++) {
        double diff = get_element(y, row_count, 0) - get_element(y_pred, row_count, 0);
        squared_diff_sum += (diff * diff);
    }

    return (squared_diff_sum / y->rows);
}

Matrix* mean_squared_error_derivative(Matrix* y, Matrix* y_pred) {
    // Creates matrix of MSE's partial derivatives with respect to each of the predictions.
    Matrix gradient_matrix = create_matrix(y->rows, 1);
    
    for (int row_count=0; row_count < y->rows; row_count++) {
        double grad = (2.0/(y->rows)) * (get_element(y_pred, row_count, 0) - get_element(y, row_count, 0));
        set_element(&gradient_matrix, row_count, 0, grad);
    }

    Matrix* output = malloc(sizeof(Matrix));
    *output = gradient_matrix;

    return output;
}

double mean_absolute_error(Matrix* y, Matrix* y_pred) {
    double abs_diff_sum = 0;
    for (int row_count=0; row_count < y->rows; row_count++) {
        double diff = get_element(y, row_count, 0) - get_element(y_pred, row_count, 0);
        abs_diff_sum += fabs(diff);
    }

    return (abs_diff_sum / y->rows);
}

Matrix* mean_absolute_error_derivative(Matrix* y, Matrix* y_pred) {
    // Creates matrix of MAE's partial derivatives with respect to each of the predictions.
    Matrix gradient_matrix = create_matrix(y->rows, 1);

    for (int row_count=0; row_count < y->rows; row_count++) {
        double y_i = get_element(y, row_count, 0);
        double y_pred_i = get_element(y_pred, row_count, 0);

        if (y_i == y_pred_i) {
            // Technically unneeded as matrix initialised to all zeros, but want to make it explicit.
            set_element(&gradient_matrix, row_count, 0, 0); 
        }
        else {
            double grad;
            if (y_i > y_pred_i) {
                grad = -1.0 / y->rows;
            }
            else {
                grad = 1.0 / y->rows;
            }

            set_element(&gradient_matrix, row_count, 0, grad);
        }
    }

    Matrix* output = malloc(sizeof(Matrix));
    *output = gradient_matrix;

    return output;
}

double binary_cross_entropy(Matrix* y, Matrix* y_pred) {
    const double epsilon = 1e-15;
    double sum = 0;

    for (int row_count=0; row_count < y->rows; row_count++) {
        double y_i = get_element(y, row_count, 0);
        double y_pred_i = get_element(y_pred, row_count, 0);

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

    return (sum / y->rows);
}

Matrix* binary_cross_entropy_derivative(Matrix* y, Matrix* y_pred) {
    // Creates matrix of MCE's partial derivatives with respect to each of the predictions.
    const double epsilon = 1e-15;
    Matrix gradient_matrix = create_matrix(y->rows, 1);

    for (int row_count=0; row_count < y->rows; row_count++) {
        double y_i = get_element(y, row_count, 0);
        double y_pred_i = get_element(y_pred, row_count, 0);
        
        if (y_pred_i < epsilon) {
            y_pred_i = epsilon;
        }
        if (y_pred_i > 1.0 - epsilon) {
            y_pred_i = 1.0 - epsilon;
        }

        double grad = (-1.0/y->rows) * ((y_i/y_pred_i) - ((1-y_i) / (1-y_pred_i)));
        set_element(&gradient_matrix, row_count, 0, grad);
    }

    Matrix* output = malloc(sizeof(Matrix));
    *output = gradient_matrix;

    return output;
}