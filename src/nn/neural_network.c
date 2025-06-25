#include <stdlib.h>
#include "nn/neural_network.h"
#include "maths/matrix.h"
#include "maths/activation.h"

static Layer init_layer(int input_size, int output_size, const ActivationFunc* activation) {
    // Initialises a layer with weight and bias matrices having all elements as zero.
    Layer new_layer;

    new_layer.weights = create_matrix(output_size, input_size);
    new_layer.biases = create_matrix(output_size, 1);
    new_layer.activation = activation;
    new_layer.num_nodes = output_size;

    // Filling with empty matrices so no errors if they are freed before a forward pass is performed.
    new_layer.z = empty_matrix();
    new_layer.a = empty_matrix();
    new_layer.dL_dz = empty_matrix();
    new_layer.dL_dw = empty_matrix();
    new_layer.dL_db = empty_matrix();

    return new_layer;
}

Network init_neural_net(int num_layers, int input_nodes, int layer_sizes[], const ActivationFunc* activations[]) {
    // Initialises a neural network with the given number of layers, and number of nodes for each layer.
    Network new_network;
    new_network.num_layers = num_layers;
    new_network.layers = calloc(num_layers, sizeof(Layer));

    for (int i=0; i < num_layers; i++) {
        int input_size = (i==0) ? input_nodes : new_network.layers[i-1].num_nodes;

        new_network.layers[i] = init_layer(input_size, layer_sizes[i], activations[i]);
    }

    return new_network;
}

static void free_layer(Layer* layer) {
    // Freeing memory allocated to storing matrices in Layer struct
    free_matrix(&layer->weights);
    free_matrix(&layer->biases);
    free_matrix(&layer->z);
    free_matrix(&layer->a);
    free_matrix(&layer->dL_dz);
    free_matrix(&layer->dL_dw);
    free_matrix(&layer->dL_db);

    layer->num_nodes = 0;
}

void free_network(Network* net) {
    // Frees memory allocated to pointers and matrices in a Network struct and its Layer structs.
    if (net->layers != NULL) {
        for (int i=0; i < net->num_layers; i++) {
            free_layer(&net->layers[i]);
        }
        
        free(net->layers);
        net->layers = NULL;
    }

    net->num_layers = 0;
}

Matrix forward_pass(Network* net, const Matrix* input) {
    Matrix layer_in, temp, layer_out;
    layer_in = copy_matrix(input);

    // Simple feedforward process: each layer's output is calculated, and given to the next layer as 
    // input until the output layer is reached. 
    for (int i=0; i < net->num_layers; i++) {
        // The pre-activation output, z, of each layer is calculated as z = wx + b, where x is the input
        // matrix, w is the weight matrix of the layer, and b is the bias matrix of the layer.
        free_matrix(&net->layers[i].z);
        free_matrix(&net->layers[i].a);
        temp = matrix_multiplication(&net->layers[i].weights, &layer_in);
        layer_out = matrix_broadcast_addition(&temp, &net->layers[i].biases);
        net->layers[i].z = copy_matrix(&layer_out);
        free_matrix(&temp);

        apply_func(&layer_out, net->layers[i].activation->func_ptr);
        net->layers[i].a = copy_matrix(&layer_out);

        free_matrix(&layer_in);
        layer_in = layer_out;
    }

    return layer_out;
}