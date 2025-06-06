#include <stdio.h>
#include "matrix.h"

Matrix forward_pass(Matrix* input, Matrix weights[], Matrix biases[], int layers) {
    Matrix layer_in, layer_out;
    layer_in = *input;

    printf("Input:\n");
    display_matrix(&layer_in);

    // Simple feedforward process: each layer's output is calculated, and given to the next layer as 
    // input until the output layer is reached. 
    for (int i=0; i < layers; i++) {
        printf("\nWeights:\n");
        display_matrix(&weights[i]);
        printf("\nBiases:\n");
        display_matrix(&biases[i]);

        // The output, y, of each layer is calculated as y = wx + b, where x is the input matrix, w is
        // the weight matrix of the layer, and b is the bias matrix of the layer.
        layer_out = matrix_multiplication(&weights[i], &layer_in);
        layer_out = matrix_addition(&layer_out, &biases[i]);

        if (i != (layers-1)) {
            printf("\nLayer output:\n");
        }
        else {
            printf("\nResult:\n");
        }

        display_matrix(&layer_out);
        layer_in = layer_out;
    }

    return layer_in; //  Returns the input to the output layer
}

void main() {
    // Creation of matrices for preset weights, biases, and input.
    Matrix weights[2];
    weights[0] = create_matrix(2, 2);
    set_element(&weights[0], 0, 0, 4);
    set_element(&weights[0], 0, 1, 8);
    set_element(&weights[0], 1, 0, 1);
    set_element(&weights[0], 1, 1, 10);

    weights[1] = create_matrix(2, 2);
    set_element(&weights[1], 0, 0, 5);
    set_element(&weights[1], 0, 1, 2);
    set_element(&weights[1], 1, 0, 9);
    set_element(&weights[1], 1, 1, 14);

    Matrix biases[2];
    biases[0] = create_matrix(2, 1);
    set_element(&biases[0], 0, 0, 1);
    set_element(&biases[0], 1, 0, 4);

    biases[1] = create_matrix(2, 1);
    set_element(&biases[1], 0, 0, 5);
    set_element(&biases[1], 1, 0, 0);

    Matrix input = create_matrix(2, 1);
    set_element(&input, 0, 0, 3);
    set_element(&input, 1, 0, 2);

    int layers = 2; // Stores the number of hidden layers, does not count the input and output layers.
    Matrix output = forward_pass(&input, weights, biases, layers);

    // Freeing memory allocated to matrices.
    free_matrix(&input);
    free_matrix(&weights[0]);
    free_matrix(&weights[1]);
    free_matrix(&biases[0]);
    free_matrix(&biases[1]);
    free_matrix(&output);
}