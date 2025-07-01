#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io/dataset_loader.h"
#include "maths/matrix.h"

static void get_dataset_dimensions(FILE* file, int* inputs_out, int* outputs_out, int* rows_out) {
    char line[4096];

    fgets(line, 4096, file);
    // Reading comment in first row to determine the number of input and output parameters
    if (sscanf(line, "# INPUTS: %d, OUTPUTS: %d", inputs_out, outputs_out) != 2) {
        printf("Error retrieving the number of input and output parameters from dataset\n");
    }

    fgets(line, 4096, file); // Skipping header row

    int rows_count=0;
    while (fgets(line, 4096, file) != NULL) { 
        rows_count++;
    }

    *rows_out = rows_count;

    rewind(file); 
}

static void fill_matrices_from_dataset(FILE* file, Matrix* input, Matrix* expected_output) {
    char line[4096];

    // Skipping first and second lines as they are comments and headers respectively.
    fgets(line, 4096, file);
    fgets(line, 4096, file);

    int sample_index = 0;
    while (fgets(line, 4096, file) != NULL) {
        char* token = strtok(line, ",");
        // Filling the input matrix.
        for (int feature_index=0; feature_index < input->rows; feature_index++) {
            if (!token) {
                break;
            }
            set_element(input, feature_index, sample_index, atof(token));
            token = strtok(NULL, ","); 
        }

        // Filling the output matrix.
        for (int output_index=0; output_index < expected_output->rows; output_index++) {
            if (!token) {
                break;
            }
            set_element(expected_output, output_index, sample_index, atof(token));
            token = strtok(NULL, ","); 
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

    int input_rows, output_rows, samples_count;
    get_dataset_dimensions(file, &input_rows, &output_rows, &samples_count);

    *input = create_matrix(input_rows, samples_count);
    *expected_output = create_matrix(output_rows, samples_count);

    fill_matrices_from_dataset(file, input, expected_output);

    fclose(file);
}