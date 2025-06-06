#include <stdio.h>
#include "matrix.h"

typedef struct Layer {
    int nodes;
    Matrix weights;
    Matrix biases;
} Layer;

Matrix forward_pass(Matrix* input, Layer layers[], int num_layers) {
    Matrix layer_in, temp, layer_out;
    layer_in = *input;

    printf("Input:\n");
    display_matrix(&layer_in);

    // Simple feedforward process: each layer's output is calculated, and given to the next layer as 
    // input until the output layer is reached. 
    for (int i=0; i < num_layers; i++) {
        printf("\nWeights:\n");
        display_matrix(&layers[i].weights);
        printf("\nBiases:\n");
        display_matrix(&layers[i].biases);

        // The output, y, of each layer is calculated as y = wx + b, where x is the input matrix, w is
        // the weight matrix of the layer, and b is the bias matrix of the layer.
        temp = matrix_multiplication(&layers[i].weights, &layer_in);
        layer_out = matrix_addition(&temp, &layers[i].biases);
        free_matrix(&temp);

        if (i != (num_layers-1)) {
            printf("\nLayer output:\n");
        }
        else {
            printf("\nResult:\n");
        }

        display_matrix(&layer_out);

        free_matrix(&layer_in);
        layer_in = layer_out;
    }

    return layer_in; //  Returns the input to the output layer
}

void main() {
    // Creation of matrices for preset weights, biases, and input.
    Layer layers[2];

    layers[0].weights = create_matrix(2, 2);
    set_element(&layers[0].weights, 0, 0, 4);
    set_element(&layers[0].weights, 0, 1, 8);
    set_element(&layers[0].weights, 1, 0, 1);
    set_element(&layers[0].weights, 1, 1, 10);

    layers[0].biases = create_matrix(2, 1);
    set_element(&layers[0].biases, 0, 0, 1);
    set_element(&layers[0].biases, 1, 0, 4);

    layers[1].weights = create_matrix(2, 2);
    set_element(&layers[1].weights, 0, 0, 5);
    set_element(&layers[1].weights, 0, 1, 2);
    set_element(&layers[1].weights, 1, 0, 9);
    set_element(&layers[1].weights, 1, 1, 14);

    layers[1].biases = create_matrix(2, 1);
    set_element(&layers[1].biases, 0, 0, 5);
    set_element(&layers[1].biases, 1, 0, 0);

    Matrix input = create_matrix(2, 1);
    set_element(&input, 0, 0, 3);
    set_element(&input, 1, 0, 2);

    int num_layers = 2; // Stores the number of non-input layers.
    Matrix output = forward_pass(&input, layers, num_layers);

    // Freeing memory allocated to matrices.
    free_matrix(&input);
    free_matrix(&layers[0].weights);
    free_matrix(&layers[0].biases);
    free_matrix(&layers[1].weights);
    free_matrix(&layers[1].biases);
    free_matrix(&output);
}