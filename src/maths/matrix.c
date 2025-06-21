#include <stdio.h>
#include <stdlib.h>
#include "maths/matrix.h"

Matrix create_matrix(int rows, int cols) {
    // Creates a matrix with the given dimensions, with all elements initialised to 0.
    Matrix new_matrix;
    new_matrix.rows = rows;
    new_matrix.cols = cols;

    // If either dimension is less than or equal to zero, return an empty matrix.
    if (rows <= 0 || cols <= 0) {
        return empty_matrix();
    }
    else {
        new_matrix.data = calloc(rows * cols, sizeof(double));

        // Checking if memory allocation failed.
        if (new_matrix.data == NULL) {
            printf("Memory allocation failed\n");
            return empty_matrix();
        }
    }

    return new_matrix;
}

Matrix empty_matrix() {
    // Returns an empty matrix, with dimensions of 0 by 0 and with data pointer set to NULL.
    Matrix empty = {0, 0, NULL};
    return empty;
}

void free_matrix(Matrix* matrix) {
    // Frees memory allocated for a matrix.
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->rows = 0;
    matrix->cols = 0;
}

Matrix copy_matrix(const Matrix* original) {
    // Creates a deep copy of a matrix.
    Matrix copy = create_matrix(original->rows, original->cols);
    for (int row_count=0; row_count < original->rows; row_count++) {
        for (int col_count=0; col_count < original->cols; col_count++) {
            double ele = get_element(original, row_count, col_count);
            set_element(&copy, row_count, col_count, ele);
        }
    }

    return copy;
}

void set_element(Matrix* matrix, int row, int col, double data_item) {
    // Sets the value of the specified element of a matrix.
    matrix->data[(row * matrix->cols) + col] = data_item;
}

double get_element(const Matrix* matrix, int row, int col) {
    // Returns the value of the specified element of a matrix.
    return matrix->data[(row * matrix->cols) + col];
}

Matrix matrix_addition(const Matrix* matrix_a, const Matrix* matrix_b) {
    // Error handling for matrices that do not have the same dimensions.
    if (matrix_a->rows != matrix_b->rows || matrix_a->cols != matrix_b->cols) {
        printf("Incompatible dimensions for matrix addition.\n");
        return empty_matrix(); 
    }

    // Calculates and returns the resulting matrix from adding the two matrices.
    Matrix result = create_matrix(matrix_a->rows, matrix_a->cols);

    for (int row_count=0; row_count < matrix_a->rows; row_count++) {
        for (int col_count=0; col_count < matrix_a->cols; col_count++) {
            double ele_a = get_element(matrix_a, row_count, col_count);
            double ele_b = get_element(matrix_b, row_count, col_count);
            double ele_result = ele_a + ele_b;

            set_element(&result, row_count, col_count, ele_result);
        }
    }

    return result;
}

Matrix matrix_multiplication(const Matrix* matrix_a, const Matrix* matrix_b) {
    // Error handling for matrices that cannot be multiplied together. 
    if (matrix_a->cols != matrix_b->rows) {
        printf("Incompatible dimensions for matrix multiplication.\n");
        return empty_matrix();
    }

    // Calculates and returns the resulting matrix from multiplying the two matrices.
    Matrix result = create_matrix(matrix_a->rows, matrix_b->cols);

    for (int i=0; i < matrix_a->rows; i++) {
        for (int j=0; j < matrix_b->cols; j++) {
            double ele = 0;
            for (int k=0; k < matrix_a->cols; k++) {
                ele += get_element(matrix_a, i, k) * get_element(matrix_b, k, j);
            }
            set_element(&result, i, j, ele);
        }
    }

    return result;
}

Matrix matrix_scalar_multiplication(const Matrix* matrix, double multiplier) {
    // Multiplies each element in a matrix by a scalar value.
    Matrix result = create_matrix(matrix->rows, matrix->cols);
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            double ele = get_element(matrix, row_count, col_count);
            set_element(&result, row_count, col_count, ele * multiplier);
        }
    }

    return result;
}

Matrix hadamard_product(const Matrix* matrix_a, const Matrix* matrix_b) {
    // Error handling for matrices that do not have same dimensions.
    if (matrix_a->rows != matrix_b->rows || matrix_a->cols != matrix_b->cols) {
        printf("Incompatible dimensions for Hadamard product.\n");
        return empty_matrix(); 
    }

    // Calculates and returns the resulting matrix from performing the Hadamard product of two matrices.
    Matrix result = create_matrix(matrix_a->rows, matrix_a->cols);

    for (int row_count=0; row_count < matrix_a->rows; row_count++) {
        for (int col_count=0; col_count < matrix_a->cols; col_count++) {
            double ele_a = get_element(matrix_a, row_count, col_count);
            double ele_b = get_element(matrix_b, row_count, col_count);
            double ele_result = ele_a * ele_b;

            set_element(&result, row_count, col_count, ele_result);
        }
    }

    return result;
}

static Matrix broadcast(const Matrix* matrix, int rows, int cols) {
    if (matrix->rows == rows && matrix->cols == cols) {
        return copy_matrix(matrix);
    }

    Matrix result = create_matrix(rows, cols);
    for (int row_count=0; row_count < rows; row_count++) {
        for (int col_count=0; col_count < cols; col_count++) {
            int source_row = (matrix->rows == 1) ? 0 : row_count;
            int source_col = (matrix->cols == 1) ? 0 : col_count;

            double ele = get_element(matrix, source_row, source_col);
            set_element(&result, row_count, col_count, ele);
        }
    }

    return result;
}

Matrix matrix_broadcast_addition(const Matrix* matrix_a, const Matrix* matrix_b) {
    int rows_compatible = (matrix_a->rows == matrix_b->rows || matrix_a->rows == 1 || matrix_b->rows == 1);
    int cols_compatible = (matrix_a->cols == matrix_b->cols || matrix_a->cols == 1 || matrix_b->cols == 1);
    if (!rows_compatible || !cols_compatible) {
        printf("Incompatible dimensions for broadcasting");
        return empty_matrix();
    }

    // Adds two matrices by broadcasting.
    int rows = (matrix_a->rows > matrix_b->rows) ? matrix_a->rows : matrix_b->rows;
    int cols = (matrix_a->cols > matrix_b->cols) ? matrix_a->cols : matrix_b->cols;

    Matrix broadcasted_a, broadcasted_b;
    broadcasted_a = broadcast(matrix_a, rows, cols);
    broadcasted_b = broadcast(matrix_b, rows, cols);

    Matrix result = matrix_addition(&broadcasted_a, &broadcasted_b);

    free_matrix(&broadcasted_a);
    free_matrix(&broadcasted_b);
    
    return result;
}

Matrix transpose(const Matrix* matrix) {
    // Constructs and returns the transpose of the matrix.
    Matrix result = create_matrix(matrix->cols, matrix->rows);

    for (int row_count=0; row_count < matrix->rows; row_count++) {
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            double ele = get_element(matrix, row_count, col_count);
            set_element(&result, col_count, row_count, ele);
        }
    } 

    return result;
}

void apply_func(Matrix* matrix, double (*func)(double)) {
    // Applies a given function to each element in a matrix.
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            double in = get_element(matrix, row_count, col_count);
            double out = func(in);
            set_element(matrix, row_count, col_count, out);
        }
    }
}

void display_matrix(const Matrix* matrix) {
    // Displays a matrix in a more human-readable format for testing purposes.
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        printf("[ ");
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            printf("%f ", get_element(matrix, row_count, col_count));
        }
        printf("]\n");
    }
}