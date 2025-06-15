#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double* data; // Pointer to matrix data. Data is stored in a 1D array: 1st row, then 2nd row, etc.
} Matrix; // Alias for struct Matrix

// Creates a matrix with the given dimensions, with all elements initialised to 0.
Matrix create_matrix(int rows, int cols);

// Frees memory allocated to matrix that is no longer needed.
void free_matrix(Matrix* matrix);

// Sets the specified element of a matrix to the specified value.
void set_element(Matrix* matrix, int row, int col, double data_item);

// Retrieves the specified element of a matrix.
double get_element(const Matrix* matrix, int row, int col);

// Calculates and returns the resulting matrix from adding the two matrices.
Matrix matrix_addition(const Matrix* matrix_a, const Matrix* matrix_b);

// Calculates and returns the resulting matrix from multiplying the two matrices.
Matrix matrix_multiplication(const Matrix* matrix_a, const Matrix* matrix_b);

// Constructs and returns the transpose of the matrix.
Matrix transpose(const Matrix* matrix);

// Applies a given function to each element in a matrix.
void apply_func(Matrix* matrix, double (*activation)(double));

// Displays a matrix in a more human-readable format for testing purposes.
void display_matrix(const Matrix* matrix);

#endif