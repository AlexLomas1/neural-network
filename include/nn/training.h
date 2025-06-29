#ifndef TRAINING_H
#define TRAINING_H

// Forward declerations
typedef struct Network Network;
typedef struct Matrix Matrix;
typedef struct LossFunc LossFunc;
typedef struct LearningRateSchedule LearningRateSchedule;

void training_loop(Network* net, int num_epoch, const Matrix* input, const Matrix* expected_output, 
    const LossFunc* loss_func, const LearningRateSchedule* lr_schedule);

#endif