#include <math.h>
#include "activation.h"
#include "matrix.h"

static const double EULERS_NUM = 2.718281828459;

void apply_activation(Matrix* matrix, double (*activation_func)(double)) {
    // Applies a given activation function to each element in a matrix.
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            double in = get_element(matrix, row_count, col_count);
            double out = activation_func(in);
            set_element(matrix, row_count, col_count, out);
        }
    }
}

double sigmoid(double x) {
    return (1 / (1 + pow(EULERS_NUM, -x)));
}

double tanh(double x) {
    return ((pow(EULERS_NUM, x) - pow(EULERS_NUM, -x)) / (pow(EULERS_NUM, x) + pow(EULERS_NUM, -x)));
}

double ReLu(double x) {
    if (x > 0) {
        return x;
    }
    else {
        return 0;
    }
}