#include "utility.h"

struct RunData{
    TimingData timing_data;
    int final_n;
    Electron* electrons;
};

RunData multiplyRun(int init_n, int capacity, int max_t, int mode, int verbose, int block_size, int sleep_time, float delta_time);