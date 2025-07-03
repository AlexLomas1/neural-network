#include <stdio.h>
#include <string.h>
#include <time.h>
#include "io/net_config_loader.h"
#include "io/train_config_loader.h"
#include "io/dataset_loader.h"
#include "nn/neural_network.h"
#include "nn/training.h"
#include "nn/lr_schedule.h"
#include "nn/evaluation.h"
#include "maths/matrix.h"
#include "maths/loss.h"

void report_progress(int current_epoch, int epochs, double loss_val) {
    printf("[Epoch %d / %d] Loss: %f\n", current_epoch, epochs, loss_val);
}

static void load_data_paths(const char* dataset_name, char* net_config_path, char* train_config_path, 
    char* train_dataset_path, char* test_dataset_path) {
    sprintf(net_config_path, "data/%s/net_config.json", dataset_name);
    sprintf(train_config_path, "data/%s/train_config.json", dataset_name);
    sprintf(train_dataset_path, "data/%s/train.csv", dataset_name);
    sprintf(test_dataset_path, "data/%s/test.csv", dataset_name);
}

static void train_neural_net(Network* net, const char* train_dataset_path, LearningRateSchedule* lr_schedule,
    const LossFunc* loss_func, int num_epoch) {
    // Loads training dataset, trains network, and reports on duration and accuracy.
    
    Matrix input, expected_output;
    load_dataset_to_matrices(train_dataset_path, &input, &expected_output);
    
    Matrix untrained_output = forward_pass(net, &input);
    double untrained_loss = loss_func->func_ptr(&expected_output, &untrained_output);
    report_progress(0, num_epoch, untrained_loss);
    free_matrix(&untrained_output);

    int report_freq = (num_epoch >= 5) ? num_epoch / 5 : 1;

    time_t train_start = clock();

    training_loop(net, num_epoch, &input, &expected_output, loss_func, lr_schedule, &report_progress, 
        report_freq);

    time_t train_end = clock();

    double train_duration = (double)(train_end - train_start) / CLOCKS_PER_SEC;
    printf("Training completed in %.3fs.\n", train_duration);

    if (loss_func == &BCE || loss_func == &CCE) { // Classification problems
        Matrix fully_trained_output = forward_pass(net, &input);
        double accuracy = calc_accuracy(&fully_trained_output, &expected_output);
        printf("Final accuracy on training dataset: %.2f%%\n", accuracy*100);
        free_matrix(&fully_trained_output);
    }
    printf("\n");

    free_matrix(&input);
    free_matrix(&expected_output);
}

static void test_neural_net(Network* net, const char* test_dataset_path, const LossFunc* loss_func) {
    // Loads testing dataset, runs the trained network on this data, and reports on duration and accuracy.

    Matrix input, expected_output;
    load_dataset_to_matrices(test_dataset_path, &input, &expected_output);

    time_t test_start = clock();

    Matrix test_output = forward_pass(net, &input);

    time_t test_end = clock();

    double test_duration = (double)(test_end - test_start) / CLOCKS_PER_SEC;
    printf("Testing completed in %.3fs.\n", test_duration);

    double loss = loss_func->func_ptr(&expected_output, &test_output);
    printf("Loss on testing dataset: %f\n", loss);

    if (loss_func == &BCE || loss_func == &CCE) { // Classification problems
        double accuracy = calc_accuracy(&test_output, &expected_output);
        printf("Accuracy on testing dataset: %.2f%%\n", accuracy*100);
    }

    free_matrix(&input);
    free_matrix(&expected_output);
    free_matrix(&test_output);
}

int main() {
    char dataset_name[32];
    printf("Enter a dataset name (e.g. xor, iris): ");
    scanf("%s", dataset_name);
    printf("\n");

    char net_config_path[128], train_config_path[128], train_dataset_path[128], test_dataset_path[128];

    load_data_paths(dataset_name, net_config_path, train_config_path, train_dataset_path, test_dataset_path);

    // Verifying that the entered dataset is valid.
    FILE* existence_check = fopen(net_config_path, "r");
    if (!existence_check) {
        printf("\"%s\" is not a valid dataset name.\n", dataset_name);
        return 1;
    }
    fclose(existence_check);

    // Creating the network 
    Network neural_net = build_network_from_config(net_config_path);

    LearningRateSchedule lr_schedule;
    const LossFunc* loss_func;
    int num_epoch;

    extract_training_parameters(train_config_path, &loss_func, &num_epoch, &lr_schedule);

    printf("---Training---\n");
    train_neural_net(&neural_net, train_dataset_path, &lr_schedule, loss_func, num_epoch);

    printf("---Testing---\n");
    test_neural_net(&neural_net, test_dataset_path, loss_func);

    // Freeing allocated memory.
    free_network(&neural_net);

    return 0;
}