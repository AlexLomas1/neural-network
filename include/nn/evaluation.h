#ifndef EVALUATION_H
#define EVALUATION_H

#include "maths/matrix.h" // For matrix struct

// Returns the accuracy of predictions made by the neural network in a classification problem
double calc_accuracy(Matrix* output, Matrix* expected_output);

#endif