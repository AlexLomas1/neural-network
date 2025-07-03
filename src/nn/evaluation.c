#include "nn/evaluation.h"
#include "maths/matrix.h"

static Matrix argmax(Matrix* output) {
    // Converts each column into one-hot format
    Matrix predictions = create_matrix(output->rows, output->cols);

    for (int col_count=0; col_count < output->cols; col_count++) {
        double max = -1.0;
        int max_index = -1;
        for (int row_count=0; row_count < output->rows; row_count++) {
            double current_ele = get_element(output, row_count, col_count);
            if (current_ele > max) {
                max = current_ele;
                max_index = row_count;
            }
        }

        // Note: create_matrix initialises all elements to 0, so all other elements are 0.
        set_element(&predictions, max_index, col_count, 1);
    }

    return predictions;
}

double calc_accuracy(Matrix* output, Matrix* expected_output) {
    // Returns the accuracy of predictions made by the neural network in a classification problem
    int total_predictions = output->cols;
    int correct_predictions = 0;

    if (output->rows == 1) { // Binary classification
        for (int col_count=0; col_count < output->cols; col_count++) {
            double expected = get_element(expected_output, 0, col_count);
            double predicted = get_element(output, 0, col_count);

            int expected_class = (expected >= 0.5) ? 1 : 0;
            int predicted_class = (predicted >= 0.5) ? 1 : 0;

            if (expected_class == predicted_class) {
                correct_predictions++;
            }
        }
    }

    else { // Multi-class classification
        Matrix predictions = argmax(output);

        for (int col_count=0; col_count < output->cols; col_count++) {
            int correct_elements = 0;
            for (int row_count=0; row_count < output->rows; row_count++) {
                if (get_element(&predictions, row_count, col_count) == 
                    get_element(expected_output, row_count, col_count)) {
                    correct_elements++;
                }
            }

            if (correct_elements == output->rows) {
                correct_predictions++;
            }
        }

        free_matrix(&predictions);
    }

    return (double)correct_predictions / total_predictions;
}