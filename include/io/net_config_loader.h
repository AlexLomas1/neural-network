#ifndef NET_CONFIG_LOADER_H
#define NET_CONFIG_LOADER_H

#include "nn/neural_network.h" // For Network struct

// Initialises a Network struct from a .json config file 
Network build_network_from_config(const char* file_path);

#endif