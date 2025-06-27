#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nn/weight_init.h"
#include "maths/matrix.h"

const WeightInit Xavier = &xavier_initialisation;
const WeightInit He = &he_initialisation;

const double PI = 3.14159265358979323846;

static int seeded = 0;
static int stored = 0;
static double stored_val = 0.0;

static void seed_if_needed() {
    if (seeded == 0) { // Only want to set the seed once.
        srand(time(NULL)); // Setting seed for rand
        seeded = 1;
    }
}

void xavier_initialisation(Matrix* weights) {
    // This is Uniform Xavier initialisation
    seed_if_needed();

    double upper_lim = sqrt(6.0 / (weights->rows + weights->cols));
    for (int row_count=0; row_count < weights->rows; row_count++) {
        for (int col_count=0; col_count < weights->cols; col_count++) {
            // Generates a random number in range [-upper_lim, upper_lim] with uniform distribution
            double ele = 2 * upper_lim * ((double)rand() / RAND_MAX) - upper_lim;
            set_element(weights, row_count, col_count, ele);
        }
    }
}

static double rand_normal() {
    // Returns a pseudo-random number from N(0, 1), using Box-Muller transformation.
    if (stored == 1) {
        stored = 0;
        return stored_val;
    }
    
    // Shifting values to keep u1 and u2 in range (0, 1)
    double u1 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / (RAND_MAX + 2.0);

    double r = sqrt(-2 * log(u1));
    double theta = 2 * PI * u2;

    // Box-Muller generates two independent values, so second value stored to be used on next call.
    double z0 = r * cos(theta);
    double z1 = r * sin(theta);

    stored_val = z1;
    stored = 1;

    return z0;
}

void he_initialisation(Matrix* weights) {
    seed_if_needed();
    
    double std_dev = sqrt(2.0 / weights->cols);
    for (int row_count=0; row_count < weights->rows; row_count++) {
        for (int col_count=0; col_count < weights->cols; col_count++) {
            // Generates a random number from normal distribution with mean 0 and standard deviation std_dev
            double ele = rand_normal() * std_dev;
            set_element(weights, row_count, col_count, ele);
        }
    }
}