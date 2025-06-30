#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io/train_config_loader.h"
#include "io/json_config_parser_priv.h"
#include "nn/training.h"
#include "nn/lr_schedule.h"
#include "maths/loss.h"

static StepDecay extract_step_decay_param(const char* data) {
    StepDecay result;
    result.decay_factor = extract_double(data, "\"decay_factor\"");
    result.step_size = extract_int(data, "\"step_size\"");
    return result;
}

static ExpDecay extract_exp_decay_param(const char* data) {
    ExpDecay result;
    result.decay_rate = extract_double(data, "\"decay_rate\"");
    return result;
}

void extract_training_parameters(const char* file_path, const LossFunc** loss_func, int* num_epoch, 
    LearningRateSchedule* lr_schedule) {
    // Extracts training parameters from a train_config.json file

    char* file_data = read_file(file_path);

    char* loss_str = extract_string(file_data, "\"loss\"");
    *num_epoch = extract_int(file_data, "\"num_epoch\"");
    lr_schedule->base_lr = extract_double(file_data, "\"learning_rate\"");
    char* lr_schedule_type_str = extract_string(file_data, "\"lr_schedule\"");

    if (strcmp(loss_str, "MSE") == 0) {
        *loss_func = &MSE;
    }
    else if (strcmp(loss_str, "MAE") == 0) {
        *loss_func = &MAE;
    }
    else if (strcmp(loss_str, "BCE") == 0) {
        *loss_func = &BCE;
    }
    else if (strcmp(loss_str, "CCE") == 0) {
        *loss_func = &CCE;
    }
    free(loss_str);

    if (strcmp(lr_schedule_type_str, "FIXED") == 0) {
        lr_schedule->type = FIXED;
    }
    else if (strcmp(lr_schedule_type_str, "STEP_DECAY") == 0) {
        lr_schedule->type = STEP_DECAY;
        lr_schedule->param.step_decay = extract_step_decay_param(file_data);
    }
    else if (strcmp(lr_schedule_type_str, "EXP_DECAY") == 0) {
        lr_schedule->type = EXPONENTIAL_DECAY;
        lr_schedule->param.exp_decay = extract_exp_decay_param(file_data);
    }
    free(lr_schedule_type_str);

    free(file_data);
}