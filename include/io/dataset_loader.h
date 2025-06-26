#ifndef DATASET_LOADER_H
#define DATASET_LOADER_H

typedef struct Matrix Matrix;

// Populates input and expected output matrices from a .csv dataset
void load_dataset_to_matrices(const char* file_path, Matrix* input, Matrix* expected_output);

#endif