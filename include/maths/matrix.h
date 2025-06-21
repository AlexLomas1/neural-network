#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double* data; // Pointer to matrix data. Data is stored in a 1D array: 1st row, then 2nd row, etc.
} Matrix; // Alias for struct Matrix

// Creates a matrix with the given dimensions, with all elements initialised to 0.
Matrix create_matrix(int rows, int cols);

// Returns an empty matrix, with dimensions of 0 by 0 and with data pointer set to NULL.
Matrix empty_matrix();

// Frees memory allocated for a matrix.
void free_matrix(Matrix* matrix);

// Creates a deep copy of a matrix.
Matrix copy_matrix(const Matrix* original);

// Sets the value of the specified element of a matrix.
void set_element(Matrix* matrix, int row, int col, double data_item);

// Returns the value of the specified element of a matrix.
double get_element(const Matrix* matrix, int row, int col);

// Calculates and returns the resulting matrix from adding the two matrices.
Matrix matrix_addition(const Matrix* matrix_a, const Matrix* matrix_b);

// Calculates and returns the resulting matrix from multiplying the two matrices.
Matrix matrix_multiplication(const Matrix* matrix_a, const Matrix* matrix_b);

// Multiplies each element in a matrix by a scalar value.
Matrix matrix_scalar_multiplication(const Matrix* matrix, double multiplier);

// Calculates and returns the resulting matrix from performing the Hadamard product of two matrices.
Matrix hadamard_product(const Matrix* matrix_a, const Matrix* matrix_b);

// Adds two matrices by broadcasting.
Matrix matrix_broadcast_addition(const Matrix* matrix_a, const Matrix* matrix_b);

// Constructs and returns the transpose of the matrix.
Matrix transpose(const Matrix* matrix);

// Applies a given function to each element in a matrix.
void apply_func(Matrix* matrix, double (*activation)(double));

// Displays a matrix in a more human-readable format for testing purposes.
void display_matrix(const Matrix* matrix);

#endif