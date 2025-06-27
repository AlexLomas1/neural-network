#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

typedef struct Matrix Matrix; // Forward declaration

typedef void (*WeightInit)(Matrix*);

extern const WeightInit Xavier;
extern const WeightInit He;

void xavier_initialisation(Matrix* weights);

void he_initialisation(Matrix* weights);

#endif