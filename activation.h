#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct ActivationFunc {
    double (*func_ptr)(double);
    double (*derivative_ptr)(double);
} ActivationFunc;

// Custom included in tanh name to prevent conflict with tanh function in math.h
extern const ActivationFunc sigmoid; 
extern const ActivationFunc tanh_custom;
extern const ActivationFunc ReLu;

// 1 / (1 + e^{-x})
double sigmoid_func(double x);

// σ(x)(1 - σ(x)), where σ(x) is the sigmoid function
double sigmoid_derivative(double x);

// (e^{x} - e^{-x}) / (e^{x} + e^{-x})
double tanh_func(double x);

// 1 - (tanh(x))^{2}
double tanh_derivative(double x);

// max(0, x)
double ReLu_func(double x);

// 1 if x > 0, otherwise 0
double ReLu_derivative(double x);

#endif