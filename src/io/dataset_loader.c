#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io/dataset_loader.h"
#include "maths/matrix.h"

static void get_dataset_dimensions(FILE* file, int* rows_out, int* cols_out) {
    char line[4096];

    int cols_count=1; // Start at 1, as there is one more column then number of commas in one line.
    fgets(line, 4096, file);
    const char* c = line;
    while (*c) { // Repeats until end of line is reached.
        if (*c == ',') {
            cols_count++;
        }
        c++;
    }

    *cols_out = cols_count;

    rewind(file);
    fgets(line, 4096, file); // First row is purposefully skipped as that is a header row.

    int rows_count=0;
    while (fgets(line, 4096, file) != NULL) { 
        rows_count++;
    }

    *rows_out = rows_count;

    rewind(file);
}

static void fill_matrices_from_dataset(FILE* file, Matrix* input, Matrix* expected_output) {
    char line[4096];

    // Skipping first line as it is just for headers.
    fgets(line, 4096, file);

    int sample_index = 0;
    while (fgets(line, 4096, file) != NULL) {
        char* token = strtok(line, ",");
        for (int feature_index=0; feature_index < input->rows; feature_index++) {
            if (!token) {
                break;
            }
            set_element(input, feature_index, sample_index, atof(token));
            token = strtok(NULL, ","); 
        }

        if (token) {
            set_element(expected_output, 0, sample_index, atof(token));
        }

        sample_index++;
    }
}

void load_dataset_to_matrices(const char* file_path, Matrix* input, Matrix* expected_output) {
    FILE* file = fopen(file_path, "r");
    if (!file) {
        printf("Error opening dataset file\n");
        return;
    }

    int samples_count, features_count;
    get_dataset_dimensions(file, &samples_count, &features_count);

    int input_rows = features_count - 1; // Assuming 1 output feature per sample.
    int input_cols = samples_count; 

    *input = create_matrix(input_rows, input_cols);
    *expected_output = create_matrix(1, input_cols);

    fill_matrices_from_dataset(file, input, expected_output);

    fclose(file);
}