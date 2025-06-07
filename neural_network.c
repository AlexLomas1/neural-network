#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

typedef struct Layer {
    Matrix weights;
    Matrix biases;
    int num_nodes;
} Layer;

typedef struct Network {
    Layer* layers;
    int num_layers;
} Network;

Layer init_layer(int input_size, int output_size) {
    // Initialises a layer with weight and bias matrices having all elements as zero.
    Layer new_layer;
    new_layer.weights = create_matrix(output_size, input_size);
    new_layer.biases = create_matrix(output_size, 1);
    new_layer.num_nodes = output_size;

    return new_layer;
}

Network init_neural_net(int num_layers, int input_nodes, int layer_sizes[]) {
    // Initialises a neural network with the given number of layers, and number of nodes for each layer.
    Network new_network;
    new_network.num_layers = num_layers;
    new_network.layers = calloc(num_layers, sizeof(Layer));

    int input_size;
    for (int i=0; i < num_layers; i++) {
        if (i==0) {
            input_size = input_nodes;
        }
        else {
            input_size = new_network.layers[i-1].num_nodes;
        }

        new_network.layers[i] = init_layer(input_size, layer_sizes[i]);
    }

    return new_network;
}

void free_network(Network* net) {
    // Frees memory allocated to layers, weight matrices, and bias matrices in the network.
    for (int i=0; i < net->num_layers; i++) {
        free_matrix(&net->layers[i].weights);
        free_matrix(&net->layers[i].biases);
        net->layers[i].num_nodes = 0;
    }

    if (net->layers != NULL) {
        free(net->layers);
        net->layers = NULL;
    }

    net->num_layers = 0;
}

Matrix forward_pass(Network* net, Matrix* input) {
    Matrix layer_in, temp, layer_out;
    layer_in = *input;

    printf("Input:\n");
    display_matrix(&layer_in);

    // Simple feedforward process: each layer's output is calculated, and given to the next layer as 
    // input until the output layer is reached. 
    for (int i=0; i < net->num_layers; i++) {
        printf("\nWeights:\n");
        display_matrix(&net->layers[i].weights);
        printf("\nBiases:\n");
        display_matrix(&net->layers[i].biases);

        // The output, y, of each layer is calculated as y = wx + b, where x is the input matrix, w is
        // the weight matrix of the layer, and b is the bias matrix of the layer.
        temp = matrix_multiplication(&net->layers[i].weights, &layer_in);
        layer_out = matrix_addition(&temp, &net->layers[i].biases);
        free_matrix(&temp);

        if (i != (net->num_layers-1)) {
            printf("\nLayer output:\n");
        }
        else {
            printf("\nResult:\n");
        }

        display_matrix(&layer_out);

        free_matrix(&layer_in);
        layer_in = layer_out;
    }

    return layer_out; 
}

void main() {
    // Layer sizes and number of layers hardcoded for now. In future, these could be retrieved from a
    // seperate file.
    int layer_sizes[2];
    layer_sizes[0] = 2;
    layer_sizes[1] = 1;
    Network test_net;
    test_net = init_neural_net(2, 2, layer_sizes);

    // Setting preset weights and biases.
    set_element(&test_net.layers[0].weights, 0, 0, 4);
    set_element(&test_net.layers[0].weights, 0, 1, 8);
    set_element(&test_net.layers[0].weights, 1, 0, 1);
    set_element(&test_net.layers[0].weights, 1, 1, 10);

    set_element(&test_net.layers[0].biases, 0, 0, 1);
    set_element(&test_net.layers[0].biases, 1, 0, 4);

    set_element(&test_net.layers[1].weights, 0, 0, 5);
    set_element(&test_net.layers[1].weights, 0, 1, 2);

    set_element(&test_net.layers[1].biases, 0, 0, 5);

    Matrix input = create_matrix(2, 1);
    double in1, in2;
    printf("Enter two input values: ");
    scanf("%lf %lf", &in1, &in2);
    set_element(&input, 0, 0, in1);
    set_element(&input, 1, 0, in2);

    Matrix output = forward_pass(&test_net, &input);

    // Freeing allocated memory.
    free_network(&test_net);
    free_matrix(&input);
    free_matrix(&output);
}