#include <stdio.h>
#include <stdlib.h>
#include "nn/neural_network.h"
#include "maths/matrix.h"
#include "maths/activation.h"

Layer init_layer(int input_size, int output_size, const ActivationFunc* activation) {
    // Initialises a layer with weight and bias matrices having all elements as zero.
    Layer new_layer;

    new_layer.weights = malloc(sizeof(Matrix));
    *new_layer.weights = create_matrix(output_size, input_size);
    new_layer.biases = malloc(sizeof(Matrix));
    *new_layer.biases = create_matrix(output_size, 1);
    new_layer.activation = activation;
    new_layer.num_nodes = output_size;

    new_layer.z = malloc(sizeof(Matrix));
    new_layer.a = malloc(sizeof(Matrix));

    return new_layer;
}

Network init_neural_net(int num_layers, int input_nodes, int layer_sizes[], const ActivationFunc* activations[]) {
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

        new_network.layers[i] = init_layer(input_size, layer_sizes[i], activations[i]);
    }

    return new_network;
}

void free_network(Network* net) {
    // Frees memory allocated to layers, weight matrices, and bias matrices in the network.
    if (net->layers != NULL) {
        for (int i=0; i < net->num_layers; i++) {
            // Freeing matrices which have pointers to them stored in layer struct.
            free_matrix(net->layers[i].weights);
            free_matrix(net->layers[i].biases);
            free_matrix(net->layers[i].z);
            free_matrix(net->layers[i].a);
            
            // Freeing memory allocated for storing these pointers.
            free(net->layers[i].weights);
            free(net->layers[i].biases);
            free(net->layers[i].z);
            free(net->layers[i].a);
            net->layers[i].num_nodes = 0;
        }
        free(net->layers);
        net->layers = NULL;
    }

    net->num_layers = 0;
}

Matrix* forward_pass(Network* net, Matrix* input) {
    Matrix layer_in, temp, layer_out;
    layer_in = copy_matrix(input);

    printf("Input:\n");
    display_matrix(&layer_in);

    // Simple feedforward process: each layer's output is calculated, and given to the next layer as 
    // input until the output layer is reached. 
    for (int i=0; i < net->num_layers; i++) {
        printf("\nWeights:\n");
        display_matrix(net->layers[i].weights);
        printf("\nBiases:\n");
        display_matrix(net->layers[i].biases);

        // The pre-activation output, z, of each layer is calculated as z = wx + b, where x is the input
        // matrix, w is the weight matrix of the layer, and b is the bias matrix of the layer.
        temp = matrix_multiplication(net->layers[i].weights, &layer_in);
        layer_out = matrix_addition(&temp, net->layers[i].biases);
        *net->layers[i].z = copy_matrix(&layer_out);
        free_matrix(&temp);

        printf("\nLayer output pre-activation:\n");
        display_matrix(&layer_out);

        apply_func(&layer_out, net->layers[i].activation->func_ptr);
        *net->layers[i].a = copy_matrix(&layer_out);

        if (i != (net->num_layers-1)) {
            printf("\nLayer output post-activation:\n");
        }
        else {
            printf("\nResult:\n");
        }

        display_matrix(&layer_out);

        free_matrix(&layer_in);
        layer_in = layer_out;
    }

    Matrix* result = malloc(sizeof(Matrix));
    *result = layer_out;

    return result; 
}