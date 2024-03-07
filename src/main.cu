#include <cuda_runtime.h>
#include <stdio.h>
#include <boost/lexical_cast.hpp>
#include "simp_simulation.h"
#include "global_gravity_simulation.h"
#include "multiply_simulation.h"
#include "test.h"
#include "pic_simulation.h"


int main(int argc, char **argv) {
    if (!strcmp(argv[1], "test")){
        runTests();
        return 0;
    }

    int verbose = boost::lexical_cast<int>(argv[2]);
    int init_n = boost::lexical_cast<int>(argv[3]);
    int max_t = boost::lexical_cast<int>(argv[4]);
    int block_size = boost::lexical_cast<int>(argv[5]);
    auto start = start_cpu_timer();
    if (!strcmp(argv[1], "0")) {
        simpSimulationRun(init_n);
    }
    else if (!strcmp(argv[1], "1")) {
        globalGravityRun(init_n, max_t, false, verbose);
    }
    else if (!strcmp(argv[1], "2")) {
        globalGravityRun(init_n, max_t, true, verbose);
    }
    else {
        int max_n = boost::lexical_cast<int>(argv[6]);
        int sleep_time = boost::lexical_cast<int>(argv[7]);
        if (!strcmp(argv[1], "3")){ //Normal
            multiplyRun(init_n, max_n, max_t, 0, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "4")){ //Huge
            multiplyRun(init_n, max_n, max_t, 1, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "5")){ //Static
            multiplyRun(init_n, max_n, max_t, 2, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "6")){ //Static advanced
            multiplyRun(init_n, max_n, max_t, 3, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "7")){ // Normal full
            multiplyRun(init_n, max_n, max_t, 4, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "8")){ // GPU Iterate with barrier using global memory
            multiplyRun(init_n, max_n, max_t, 5, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "9")){ // GPU Iterate with barrier using global memory with cooperative
            multiplyRun(init_n, max_n, max_t, 5, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "10")){ // GPU Iterate with barrier using global memory organised
            multiplyRun(init_n, max_n, max_t, 6, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "11")){ // GPU Iterate with barrier using multi block sync
            multiplyRun(init_n, max_n, max_t, 7, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "12")){ // GPU Iterate with barrier using multi block sync
            multiplyRun(init_n, max_n, max_t, 8, verbose, block_size, sleep_time);
        }
        else if (!strcmp(argv[1], "13")){ // PIC
            runPIC(init_n, max_n, max_t, verbose, block_size);
        }
    }
    double time = end_cpu_timer(start);
    printf("CPU time of program: %f ms\n", time);
}