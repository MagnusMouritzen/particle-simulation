#pragma once
#include <algorithm>
#include <cmath>
#include "pic.h"

void runBenchmark();

void runUnitTest(int init_n, int max_n, int max_t, int poisson_timestep, int verbose, int block_size, int sleep_time);