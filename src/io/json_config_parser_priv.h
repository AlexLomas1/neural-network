#ifndef JSON_CONFIG_PARSER_PRIV_H
#define JSON_CONFIG_PARSER_PRIV_H

// Reads a file, and returns the file contents as a string.
char* read_file(const char* file_path);

// Finds first occurance of param_name, and returns the value following it as an integer.
int extract_int(const char* data, const char* param_name);

// Finds first occurance of param_name, and returns the value following it as an double.
double extract_double(const char* data, const char* param_name);

// Finds first occurance of param_name, and returns the value of the string following it.
char* extract_string(const char* data, const char* param_name);

#endif