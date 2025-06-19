#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "maths/matrix.h" // For Matrix struct and matrix operations

// Forward declaration of struct defined in activation.h
typedef struct ActivationFunc ActivationFunc;

typedef struct Layer {
    Matrix weights;
    Matrix biases;
    const ActivationFunc* activation;
    int num_nodes;

    Matrix z; // Pre-activation output of layer
    Matrix a; // Post-activation output of layer

    Matrix dL_dz; // Matrix of partial derivative of loss with respect to z (z = wx + b)
    Matrix dL_dw; // Matrix of partial derivative of loss with respect to weights
    Matrix dL_db; // Matrix of partial derivative of loss with respect to biases
} Layer;

typedef struct Network {
    Layer* layers;
    int num_layers;
} Network;

// Initialises a neural network with the given number of layers, and number of nodes for each layer.
Network init_neural_net(int num_layers, int input_nodes, int layer_sizes[], const ActivationFunc* activations[]);

// Frees memory allocated to pointers and matrices in a Network struct and its Layer structs.
void free_network(Network* net);

// Performs forward pass of data through neural net: each layer's output is calculated, and given to the
// next layer as input until the output layer is reached.
Matrix forward_pass(Network* net, const Matrix* input);

#endif