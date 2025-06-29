#include <math.h>
#include "nn/lr_schedule.h"

static double step_decay(double base_lr, int current_epoch, const StepDecay* step_decay_info) {
    int steps = current_epoch / step_decay_info->step_size;
    return base_lr * pow(step_decay_info->decay_factor, steps);
}

static double exp_decay(double base_lr, int current_epoch, const ExpDecay* exp_decay_info) {
    return base_lr * exp(-exp_decay_info->decay_rate * current_epoch);
}

double update_learning_rate(int current_epoch, const LearningRateSchedule* schedule) {
    switch (schedule->type) {
        case STEP_DECAY:
            return step_decay(schedule->base_lr, current_epoch, &schedule->param.step_decay);

        case EXPONENTIAL_DECAY:
            return exp_decay(schedule->base_lr, current_epoch, &schedule->param.exp_decay);

        default: // If FIXED or an unknown schedule type
            return schedule->base_lr;
    }
}