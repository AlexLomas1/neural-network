#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

// Forward declerations of structs defined elsewhere
typedef struct ActivationFunc ActivationFunc; // Defined in activation.h
typedef struct Matrix Matrix; // Defined in matrix.h

typedef struct Layer {
    Matrix* weights;
    Matrix* biases;
    const ActivationFunc* activation;
    int num_nodes;
} Layer;

typedef struct Network {
    Layer* layers;
    int num_layers;
} Network;

// Initialises a layer with weight and bias matrices having all elements as zero.
Layer init_layer(int input_size, int output_size, const ActivationFunc* activation);

// Initialises a neural network with the given number of layers, and number of nodes for each layer.
Network init_neural_net(int num_layers, int input_nodes, int layer_sizes[], const ActivationFunc* activations[]);

// Frees memory allocated to layers, weight matrices, and bias matrices in the network.
void free_network(Network* net);

// Performs forward pass of data through neural net: each layer's output is calculated, and given to the
// next layer as input until the output layer is reached.
Matrix* forward_pass(Network* net, Matrix* input);

#endif