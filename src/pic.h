#include "utility.h"
#include "electron.h"
#include "random.h"
#include "particle_move.h"
#include "cell.h"
RunData runPIC(int init_n, int capacity, int poisson_steps, int poisson_timestep, int mode, int verbose, int block_size, int sleep_time_ns, float split_chance, float remove_chance);