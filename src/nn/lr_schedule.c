#include <math.h>
#include "nn/lr_schedule.h"

static double step_decay(double base_lr, int current_epoch, const StepDecay* step_decay_info) {
    double new_lr;
    int steps = floor(current_epoch / step_decay_info->step_size);

    new_lr = base_lr * pow(step_decay_info->decay_factor, steps);
    return new_lr;
}

static double exp_decay(double base_lr, int current_epoch, const ExpDecay* exp_decay_info) {
    return base_lr * exp(-exp_decay_info->decay_rate * current_epoch);
}

double update_learning_rate(double learning_rate, int current_epoch, const LearningRateSchedule* schedule) {
    switch (schedule->type) {
        case FIXED:
            return learning_rate;

        case STEP_DECAY:
            return step_decay(schedule->base_lr, current_epoch, &schedule->param.step_decay);

        case EXPONENTIAL_DECAY:
            return exp_decay(schedule->base_lr, current_epoch, &schedule->param.exp_decay);

        default:
            return learning_rate;
    }
}