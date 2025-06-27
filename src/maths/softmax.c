#include <math.h>
#include <stddef.h>
#include "maths/softmax.h"
#include "maths/activation.h"
#include "maths/matrix.h"

// Softmax is a special case activation function, in that it is not element-wise. NULL attributes as the
// softmax functions are not the correct type for the ActivationFunc attributes.
const ActivationFunc softmax = {NULL, NULL};

Matrix softmax_func(const Matrix* x) {
    Matrix result = create_matrix(x->rows, x->cols);

    // Softmax is applied to each column (sample) independently.
    for (int col_count=0; col_count < x->cols; col_count++) {
        // Finding the max value in the column
        double max_val = get_element(x, 0, col_count);
        for (int row_count=1; row_count < x->rows; row_count++) {
            double ele = get_element(x, row_count, col_count);
            if (ele > max_val) {
                max_val = ele;
            }
        }

        double exp_sum = 0.0;
        for (int row_count=0; row_count < x->rows; row_count++) {
            double ele = get_element(x, row_count, col_count);
            double exp_ele = exp(ele - max_val);
            set_element(&result, row_count, col_count, exp_ele);
            exp_sum += exp_ele;
        }

        for (int row_count=0; row_count < x->rows; row_count++) {
            double result_ele = get_element(&result, row_count, col_count);
            set_element(&result, row_count, col_count, result_ele / exp_sum);
        }
    }

    return result;
}

Matrix softmax_derivative(const Matrix* x, const Matrix* loss_deriv) {
    Matrix gradient_matrix = create_matrix(x->rows, x->cols);

    for (int col_count=0; col_count < x->cols; col_count++) {
        for (int i=0; i < x->rows; i++) {
            double s_i = get_element(x, i, col_count);
            double grad_sum = 0.0;

            for (int j=0; j < x->rows; j++) {
                double s_j = get_element(x, j, col_count);
                double dL_da_j = get_element(loss_deriv, j, col_count);

                double jacobian_i_j;
                if (i == j) {
                    jacobian_i_j = s_i * (1.0 - s_j);
                }
                else {
                    jacobian_i_j = -s_i * s_j;
                }

                grad_sum += jacobian_i_j * dL_da_j;
            }

            set_element(&gradient_matrix, i, col_count, grad_sum);
        }
    }

    return gradient_matrix;
}