#include "nn/training.h"
#include "nn/neural_network.h"
#include "nn/lr_schedule.h"
#include "maths/matrix.h"
#include "maths/activation.h"
#include "maths/softmax.h"
#include "maths/loss.h"

static Matrix mean_rows(const Matrix* matrix) {
    // Returns a column vector, with each element as the mean of the corresponding row in the input matrix.
    Matrix result = create_matrix(matrix->rows, 1);
    for (int row_count=0; row_count < matrix->rows; row_count++) {
        double sum = 0.0;
        for (int col_count=0; col_count < matrix->cols; col_count++) {
            sum += get_element(matrix, row_count, col_count);
        }
        double avg = sum / matrix->cols;

        set_element(&result, row_count, 0, avg);
    }

    return result;
}

static void backpropagation(Network* net, const Matrix* input, const Matrix* loss_deriv) { 
    Layer* output_layer= &net->layers[net->num_layers-1];

    // Freeing derivative matrices before overwriting
    free_matrix(&output_layer->dL_dz);
    free_matrix(&output_layer->dL_dw);
    free_matrix(&output_layer->dL_db);

    // dL_dz = dL_da * da_dz
    if (output_layer->activation == &softmax) {
        output_layer->dL_dz = softmax_derivative(&output_layer->a, loss_deriv);
    }
    else {
        Matrix dL_da = copy_matrix(loss_deriv);
        Matrix da_dz = copy_matrix(&output_layer->z);
        apply_func(&da_dz, output_layer->activation->derivative_ptr);
        output_layer->dL_dz = hadamard_product(&dL_da, &da_dz);
        free_matrix(&dL_da);
        free_matrix(&da_dz);
    }

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
    output_layer->dL_db = mean_rows(&output_layer->dL_dz);

    for (int layer_count=net->num_layers-2; layer_count >= 0; layer_count--) {
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
        curr_layer->dL_db = mean_rows(&curr_layer->dL_dz);
    }
}

static void gradient_descent(Network* net, double learning_rate) {
    // Updates the weights and biases of each layer based on gradients calculated from backpropagation
    // and the learning rate.
    for (int layer_count=0; layer_count < net->num_layers; layer_count++) {
        Layer* curr_layer = &net->layers[layer_count];

        Matrix diff = matrix_scalar_multiplication(&curr_layer->dL_dw, -learning_rate);
        Matrix new_weights = matrix_addition(&curr_layer->weights, &diff);
        free_matrix(&curr_layer->weights);
        free_matrix(&diff);
        curr_layer->weights = copy_matrix(&new_weights);
        free_matrix(&new_weights);

        diff = matrix_scalar_multiplication(&curr_layer->dL_db, -learning_rate);
        Matrix new_biases = matrix_addition(&curr_layer->biases, &diff);
        free_matrix(&curr_layer->biases);
        free_matrix(&diff);
        curr_layer->biases = copy_matrix(&new_biases);
        free_matrix(&new_biases);
    }
}

static void train_step(Network* net, const Matrix* input, const Matrix* expected_output, 
    const LossFunc* loss_func, double learning_rate) {
    // Performs one training step: forward pass, loss calculation, backward pass, and parameter updates.
    Matrix output = forward_pass(net, input);

    Matrix loss_deriv = loss_func->derivative_ptr(expected_output, &output);
    backpropagation(net, input, &loss_deriv);
    
    free_matrix(&output);
    free_matrix(&loss_deriv);

    gradient_descent(net, learning_rate);
}

void training_loop(Network* net, int num_epoch, const Matrix* input, const Matrix* expected_output, 
    const LossFunc* loss_func, const LearningRateSchedule* lr_schedule) {

    double learning_rate = lr_schedule->base_lr;
    for (int epoch_count=0; epoch_count < num_epoch; epoch_count++) {
        train_step(net, input, expected_output, loss_func, learning_rate);
        learning_rate = update_learning_rate(epoch_count, lr_schedule);
    }
}