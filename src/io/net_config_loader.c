#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io/net_config_loader.h"
#include "io/json_config_parser_priv.h"
#include "nn/neural_network.h"
#include "nn/weight_init.h"
#include "maths/activation.h"
#include "maths/softmax.h"

static void extract_layer(const char* file, int layer_num, int* layer_size_out, 
    const ActivationFunc** activation_out, WeightInit* weight_init_fn_out) {
    // Extracts the number of nodes and the activation function for the specified layer
    char* pos = strstr(file, "\"layers\"");
    
    int curr_layer=0;
    while (strstr(pos, "\"nodes\"") != NULL) {
        // Advance through each instances of "nodes" (which occurs once for each layer) until reaching
        // the specified layer (0-indexed).
        pos = strstr(pos, "\"nodes\""); 

        if (curr_layer == layer_num) {
            *layer_size_out = extract_int(pos, "\"nodes\"");
            char* activation_str = extract_string(pos, "\"activation\"");

            // Maps activation name to pointer to the corresponding ActivationFunc constant
            if (strcmp(activation_str, "sigmoid") == 0) {
                *activation_out = &sigmoid;
            }
            else if (strcmp(activation_str, "tanh") == 0) {
                *activation_out = &tanh_custom;
            }
            else if (strcmp(activation_str, "ReLu") == 0) {
                *activation_out = &ReLu;
            }
            else if (strcmp(activation_str, "softmax") == 0) {
                *activation_out = &softmax;
            }
            free(activation_str);

            char* weight_init_str = extract_string(pos, "\"weight_init\"");

            if (strcmp(weight_init_str, "Xavier") == 0) {
                *weight_init_fn_out = Xavier;
            }
            else if (strcmp(weight_init_str, "He") == 0) {
                *weight_init_fn_out = He;
            }
            free(weight_init_str);

            return;
        }
        else {
            pos++; // Advance the position pointer to prevent being stuck on first instance of "nodes"
            curr_layer++;
        }        
    }
}

Network build_network_from_config(const char* file_path) {
    // Builds and returns a Network struct defined by a JSON config file.
    char* file_data = read_file(file_path);

    int input_nodes = extract_int(file_data, "\"input_nodes\"");
    int num_layers = extract_int(file_data, "\"num_layers\"");

    int layer_sizes[num_layers];
    const ActivationFunc* activations[num_layers];
    WeightInit weight_init_fns[num_layers];

    for (int i=0; i < num_layers; i++) {
        int curr_layer_size;
        const ActivationFunc* curr_activation_func;
        WeightInit weight_init_fn;
        extract_layer(file_data, i, &curr_layer_size, &curr_activation_func, &weight_init_fn);

        layer_sizes[i] = curr_layer_size;
        activations[i] = curr_activation_func;
        weight_init_fns[i] = weight_init_fn;
    }

    free(file_data);

    return init_neural_net(num_layers, input_nodes, layer_sizes, activations, weight_init_fns);
}