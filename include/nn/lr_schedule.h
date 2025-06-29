#ifndef LR_SCHEDULE_H
#define LR_SCHEDULE_H

typedef enum ScheduleType {
    FIXED,
    STEP_DECAY,
    EXPONENTIAL_DECAY
} ScheduleType;

typedef struct StepDecay {
    double decay_factor;
    int step_size;
} StepDecay;

typedef struct ExpDecay {
    double decay_rate;
} ExpDecay;

typedef struct LearningRateSchedule {
    ScheduleType type;
    double base_lr;

    union {
        StepDecay step_decay;
        ExpDecay exp_decay;
    } param;
} LearningRateSchedule;

double update_learning_rate(int current_epoch, const LearningRateSchedule* schedule);

#endif