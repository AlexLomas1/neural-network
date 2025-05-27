#include <stdio.h>
#include <stdlib.h>

struct Matrix {
    int rows;
    int cols;
    double* data; // Data stored in flat 1D array: 1st row, then 2nd row, then 3rd row, etc.
};

typedef struct Matrix Matrix; // So can just use Matrix, instead of always using struct Matrix

Matrix create_matrix(int rows, int cols) {
    Matrix new_matrix;
    new_matrix.rows = rows;
    new_matrix.cols = cols;
    new_matrix.data = calloc(rows * cols, sizeof(double));

    return new_matrix;
}

void free_matrix(Matrix* matrix) {
    if (matrix->data != NULL) {
        free(matrix->data);
        matrix->data = NULL;
    }
    matrix->rows = 0;
    matrix->cols = 0;
}

void set_element(Matrix* matrix, int row, int col, double data_item) {
    matrix->data[(row * matrix->cols) + col] = data_item;
}

double get_element(Matrix* matrix, int row, int col) {
    return matrix->data[(row * matrix->cols) + col];
}

Matrix matrix_addition(Matrix* a, Matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        // Error handling TBA
    }

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

Matrix matrix_multiplication(Matrix* a, Matrix* b) {
    if (a->cols != b->rows) {
        // Error handling TBA
    }

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

Matrix transpose(Matrix* matrix) {
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

void display_matrix(Matrix* matrix) {
    // This function is just for visualising matrices for testing purposes. May be removed later
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        printf("[ ");
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            printf("%f", get_element(matrix, row_count, col_count));
            printf(" ");
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