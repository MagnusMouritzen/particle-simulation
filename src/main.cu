#include <cuda_runtime.h>
#include <stdio.h>
#include <boost/lexical_cast.hpp>
#include "test.h"
#include "pic.h"


int main(int argc, char **argv) {
    // Runs benchmarks defiend in test.cu
    if (!strcmp(argv[1], "bench")){
        runBenchmark();
        return 0;
    }

    int verbose = boost::lexical_cast<int>(argv[2]);
    int init_n = boost::lexical_cast<int>(argv[3]);
    int max_t = boost::lexical_cast<int>(argv[4]);
    int block_size = boost::lexical_cast<int>(argv[5]);
    auto start = start_cpu_timer();

    int max_n = boost::lexical_cast<int>(argv[6]);
    int sleep_time = boost::lexical_cast<int>(argv[7]);
    int poisson_timestep = boost::lexical_cast<int>(argv[8]);
    
    // Dynamic
    if (!strcmp(argv[1], "30")){
        runPIC(init_n, max_n, max_t, poisson_timestep, 0, verbose, block_size, sleep_time);
    }
    // CPUSynch
    else if (!strcmp(argv[1], "31")){
        runPIC(init_n, max_n, max_t, poisson_timestep, 1, verbose, block_size, sleep_time);
    }
    // Naive
    else if (!strcmp(argv[1], "32")){
        runPIC(init_n, max_n, max_t, poisson_timestep, 2, verbose, block_size, sleep_time);
    }
    // Dynamic Old
    else if (!strcmp(argv[1], "33")){
        runPIC(init_n, max_n, max_t, poisson_timestep, 3, verbose, block_size, sleep_time);
    }
    // Runs test defined in test.cu
    else if (!strcmp(argv[1], "test")){
        runUnitTest(init_n, max_n, max_t, poisson_timestep, verbose, block_size, sleep_time);
    }
    double time = end_cpu_timer(start);
    printf("CPU time of program: %f ms\n", time);
}