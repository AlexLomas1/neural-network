#ifndef TRAIN_CONFIG_LOADER_H
#define TRAIN_CONFIG_LOADER_H

typedef struct LossFunc LossFunc;
typedef struct LearningRateSchedule LearningRateSchedule;

// Extracts training parameters from a train_config.json file
void extract_training_parameters(const char* file_path, const LossFunc** loss_func, int* num_epoch, 
    LearningRateSchedule* lr_schedule);

#endif