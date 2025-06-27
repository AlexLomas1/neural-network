#include <stdio.h>
#include "io/nn_config_loader.h"
#include "io/dataset_loader.h"
#include "nn/neural_network.h"
#include "nn/training.h"
#include "maths/matrix.h"
#include "maths/activation.h"
#include "maths/softmax.h"
#include "maths/loss.h"

int main() {
    // Creating the network 
    Network test_net = build_network_from_config("data/nn_config/xor_nn.json");

    // Setting initial weights and biases.
    set_element(&test_net.layers[0].weights, 0, 0, 0.3);
    set_element(&test_net.layers[0].weights, 0, 1, -0.6);
    set_element(&test_net.layers[0].weights, 1, 0, 0.75);
    set_element(&test_net.layers[0].weights, 1, 1, -0.9);

    set_element(&test_net.layers[0].biases, 0, 0, 0);
    set_element(&test_net.layers[0].biases, 1, 0, 0);

    set_element(&test_net.layers[1].weights, 0, 0, 0.4);
    set_element(&test_net.layers[1].weights, 0, 1, -0.7);

    set_element(&test_net.layers[1].biases, 0, 0, 0);

    Matrix input_train, expected_output_train;
    load_dataset_to_matrices("data/datasets/xor_train.csv", &input_train, &expected_output_train);

    // 1000 epochs is more than is necessary for 100% accuracy, but gives greater probability calibration
    training_loop(&test_net, 1000, &input_train, &expected_output_train, &BCE);

    free_matrix(&input_train);
    free_matrix(&expected_output_train);

    Matrix input_test, expected_output_test;
    load_dataset_to_matrices("data/datasets/xor_test.csv", &input_test, &expected_output_test);

    Matrix test_output = forward_pass(&test_net, &input_test);
    double loss = BCE.func_ptr(&expected_output_test, &test_output);

    printf("Test output:\n");
    display_matrix(&test_output);
    printf("Expected test output:\n");
    display_matrix(&expected_output_test);
    printf("Loss: %f \n", loss);

    // Freeing allocated memory.
    free_matrix(&input_test);
    free_matrix(&expected_output_test);
    free_matrix(&test_output);
    free_network(&test_net); 
}