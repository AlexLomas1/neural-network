#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "io/json_config_parser_priv.h"

char* read_file(const char* file_path) {
    // Reads a file, and returns the file contents as a string.
    FILE* file = fopen(file_path, "r");
    if (!file) {
        printf("Error opening network config file\n");
        return NULL;
    }

    fseek(file, 0, SEEK_END); // Moves file pointer to the end of the file.
    long size = ftell(file); // Gets the byte count for the entire file.
    rewind(file); // Return to start of file

    char* buffer = malloc(size + 1); // + 1 to store the null terminator as well 
    fread(buffer, 1, size, file); // Stores the contents of the file in buffer
    buffer[size] = '\0'; 

    fclose(file);
    return buffer;
}

int extract_int(const char* data, const char* param_name) {
    // Finds first occurance of param_name, and returns the value following it as an integer.
    char* pos = strstr(data, param_name); 
    pos = strchr(pos, ':'); 
    return atoi(pos+1); 
}

double extract_double(const char* data, const char* param_name) {
    // Finds first occurance of param_name, and returns the value following it as an double.
    char* pos = strstr(data, param_name);
    pos = strchr(pos, ':');
    return atof(pos+1); 
}

char* extract_string(const char* data, const char* param_name) {
    // Finds first occurance of param_name, and returns the value of the string following it.
    char* pos = strstr(data, param_name);
    pos = strchr(pos, ':');
    pos = strchr(pos, '\"'); 
    pos++; // First character of string value
    
    char* end = strchr(pos, '\"'); // End of string value

    int length = end-pos; 

    char* value = malloc(length+1);
    strncpy(value, pos, length); // Copies the string value into the value variable.
    value[length] = '\0'; 
    return value;
}