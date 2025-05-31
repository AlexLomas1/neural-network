#include <stdio.h>
#include <stdlib.h>

typedef struct Matrix {
    int rows;
    int cols;
    double* data; // Pointer to matrix data. Data is stored in a 1D array: 1st row, then 2nd row, etc.
} Matrix; // Alias for struct Matrix

Matrix create_matrix(int rows, int cols) {
    // Creates a matrix with the given dimensions, with all elements initialised to 0.
    Matrix new_matrix;
    new_matrix.rows = rows;
    new_matrix.cols = cols;

    // If either dimension is less than or equal to zero, return an empty matrix.
    if (rows <= 0 || cols <= 0) {
        new_matrix.rows = 0;
        new_matrix.cols = 0;
        new_matrix.data = NULL;
    }
    else {
        new_matrix.data = calloc(rows * cols, sizeof(double));

        // Checking if memory allocation failed.
        if (new_matrix.data == NULL) {
            printf("Memory allocation failed\n");
            new_matrix.rows = 0;
            new_matrix.cols = 0;
        }
    }

    return new_matrix;
}

void free_matrix(Matrix* matrix) {
    // Frees memory allocated to matrix that is no longer needed.
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->rows = 0;
    matrix->cols = 0;
}

void set_element(Matrix* matrix, int row, int col, double data_item) {
    // Sets the specified element of a matrix to the specified value.
    matrix->data[(row * matrix->cols) + col] = data_item;
}

double get_element(const Matrix* matrix, int row, int col) {
    // Retrieves the specified element of a matrix.
    return matrix->data[(row * matrix->cols) + col];
}

Matrix matrix_addition(const Matrix* a, const Matrix* b) {
    // Error handling for matrices that do not have the same dimensions.
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Incompatible dimensions for matrix addition.\n");
        return create_matrix(0, 0); // Empty matrix returned to indicate error.
    }

    // Calculates and returns the resulting matrix from adding the two matrices.
    Matrix result = create_matrix(a->rows, a->cols);

    for (int row_count=0; row_count < a->rows; row_count++) {
        for (int col_count=0; col_count < a->cols; col_count++) {
            double ele_a = get_element(a, row_count, col_count);
            double ele_b = get_element(b, row_count, col_count);
            double ele_result = ele_a + ele_b;

            set_element(&result, row_count, col_count, ele_result);
        }
    }

    return result;
}

Matrix matrix_multiplication(const Matrix* a, const Matrix* b) {
    // Error handling for matrices that cannot be multiplied together. 
    if (a->cols != b->rows) {
        printf("Incompatible dimensions for matrix multiplication.\n");
        return create_matrix(0, 0); // Empty matrix returned to indicate error.
    }

    // Calculates and returns the resulting matrix from multiplying the two matrices.
    Matrix result = create_matrix(a->rows, b->cols);

    for (int i=0; i < a->rows; i++) {
        for (int j=0; j < b->cols; j++) {
            double ele = 0;
            for (int k=0; k < a->cols; k++) {
                ele += get_element(a, i, k) * get_element(b, k, j);
            }
            set_element(&result, i, j, ele);
        }
    }

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

void display_matrix(const Matrix* matrix) {
    // This function is just for visualising matrices for testing purposes. May be removed later
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        printf("[ ");
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            printf("%f ", get_element(matrix, row_count, col_count));
        }
        printf("]\n");
    }
}

int main() {
    // Just a test sequence, will be removed later.
    Matrix matrix_a = create_matrix(2, 2);
    Matrix matrix_b = create_matrix(2, 2);
    Matrix result;

    set_element(&matrix_a, 0, 0, 5);
    set_element(&matrix_a, 0, 1, 7);
    set_element(&matrix_a, 1, 0, 13);
    set_element(&matrix_a, 1, 1, 4);

    set_element(&matrix_b, 0, 0, 8);
    set_element(&matrix_b, 0, 1, 0);
    set_element(&matrix_b, 1, 0, 9);
    set_element(&matrix_b, 1, 1, 27);

    result = matrix_multiplication(&matrix_a, &matrix_b);
    
    display_matrix(&matrix_a);
    printf(" \n");
    display_matrix(&matrix_b);
    printf(" \n");
    display_matrix(&result);

    free_matrix(&matrix_a);
    free_matrix(&matrix_b);
    free_matrix(&result);
}