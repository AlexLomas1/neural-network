#include <stdio.h>
#include "maths/matrix.h"
#include "maths/activation.h"
#include "maths/loss.h"
#include "nn/neural_network.h"

void backpropagation(Network* net, const Matrix* input, const Matrix* loss_deriv) { 
    Layer* output_layer= &net->layers[net->num_layers-1];

    // Freeing derivative matrices before overwriting
    free_matrix(&output_layer->dL_dz);
    free_matrix(&output_layer->dL_dw);
    free_matrix(&output_layer->dL_db);

    // dL_dz = dL_da * da_dz
    Matrix dL_da = copy_matrix(loss_deriv); // loss_func->derivative_ptr(y, y_pred)
    Matrix da_dz = copy_matrix(&output_layer->z);
    apply_func(&da_dz, output_layer->activation->derivative_ptr);
    output_layer->dL_dz = hadamard_product(&dL_da, &da_dz);
    free_matrix(&dL_da);
    free_matrix(&da_dz);

    // dL_dw = dL_dz * dz_dw
    Matrix dz_dw;
    if (net->num_layers > 1) {
        dz_dw = transpose(&net->layers[net->num_layers-2].a);
    }
    else {
        dz_dw = transpose(input);
    }
    output_layer->dL_dw = matrix_multiplication(&output_layer->dL_dz, &dz_dw);
    free_matrix(&dz_dw);

    // dL_db = dL_dz * dz_db = dL_dz * 1
    output_layer->dL_db = copy_matrix(&output_layer->dL_dz);

    for (int layer_count=net->num_layers-2; layer_count >= 0; layer_count--) {
        printf("Layer: %d\n", layer_count);
        Layer* curr_layer = &net->layers[layer_count];
        Layer* next_layer = &net->layers[layer_count+1];
        
        // Freeing derivative matrices before overwriting
        free_matrix(&curr_layer->dL_dz);
        free_matrix(&curr_layer->dL_dw);
        free_matrix(&curr_layer->dL_db);

        // dL_dz = dL_da * da_dz
        // dL_da = dL_dz{next} * dz{next}_da
        Matrix w_T = transpose(&next_layer->weights);
        Matrix dL_da = matrix_multiplication(&w_T, &next_layer->dL_dz);
        free_matrix(&w_T);
        Matrix da_dz = copy_matrix(&curr_layer->z);
        apply_func(&da_dz, curr_layer->activation->derivative_ptr);
        curr_layer->dL_dz = hadamard_product(&dL_da, &da_dz);
        free_matrix(&dL_da);
        free_matrix(&da_dz);

        // dL_dw = dL_dz * dz_dw
        if (layer_count > 0) {
            dz_dw = transpose(&net->layers[layer_count-1].a);
        }
        else {
            dz_dw = transpose(input);
        }
        curr_layer->dL_dw = matrix_multiplication(&curr_layer->dL_dz, &dz_dw);
        free_matrix(&dz_dw);

        // dL_db = dL_dz * dz_db = dL_dz * 1
        curr_layer->dL_db = copy_matrix(&curr_layer->dL_dz);
    }
    printf("Backpropagation completed, hopefully with no incompatable matrix dimensions errors\n");
}

void gradient_descent(Network* net, double learning_rate) {
    // Updates the weights and biases of each layer based on gradients calculated from backpropagation
    // and the learning rate.
    for (int layer_count=0; layer_count < net->num_layers; layer_count++) {
        Layer* curr_layer = &net->layers[layer_count];

        Matrix diff = matrix_scalar_multiplication(&curr_layer->dL_dw, -learning_rate);
        Matrix new_weights = matrix_addition(&curr_layer->weights, &diff);
        free_matrix(&curr_layer->weights);
        free_matrix(&diff);
        curr_layer->weights = copy_matrix(&new_weights);

        diff = matrix_scalar_multiplication(&curr_layer->dL_db, -learning_rate);
        Matrix new_biases = matrix_addition(&curr_layer->biases, &diff);
        free_matrix(&curr_layer->biases);
        free_matrix(&diff);
        curr_layer->biases = copy_matrix(&new_biases);

        display_matrix(&curr_layer->weights);
        printf("\n");
        display_matrix(&curr_layer->biases);
        printf("\n");
    }
}

void train_step(Network* net, const Matrix* input, const Matrix* expected_output, const LossFunc* loss_func, double learning_rate) {
    // Performs one training step: forward pass, loss calculation, backward pass, and parameter updates.
    Matrix output = forward_pass(net, input);
    printf("\n");
    printf("Correct output:\n");
    display_matrix(expected_output);
    printf("\n");

    double loss_val = loss_func->func_ptr(expected_output, &output);
    printf("Loss: %f \n", loss_val);
    Matrix loss_deriv = loss_func->derivative_ptr(expected_output, &output);
    backpropagation(net, input, &loss_deriv);
    free_matrix(&output);
    free_matrix(&loss_deriv);

    gradient_descent(net, learning_rate);
}

int main() { // test sequence for now, to be removed later
    // Layer sizes, number of layers, and activation functions hardcoded for now. In future, these could
    // be retrieved from a seperate file.
    int layer_sizes[2];
    layer_sizes[0] = 2;
    layer_sizes[1] = 1;

    const ActivationFunc* activations[2];
    activations[0] = &tanh_custom;
    activations[1] = &ReLu;

    Network test_net;
    test_net = init_neural_net(2, 2, layer_sizes, activations);

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

    Matrix input = create_matrix(2, 1); // Two features, one sample (no batching for now)
    double in1, in2;
    printf("Enter two input values: ");
    scanf("%lf %lf", &in1, &in2);
    set_element(&input, 0, 0, in1);
    set_element(&input, 1, 0, in2);

    Matrix expected_output = create_matrix(1, 1);
    set_element(&expected_output, 0, 0, 500);

    for (int i=0; i < 20; i++) {
        train_step(&test_net, &input, &expected_output, &MSE, 0.1);
    }

    // Freeing allocated memory.
    free_network(&test_net);
    free_matrix(&input);
    free_matrix(&expected_output);
}