#include <cuda_runtime.h>
#include <stdio.h>
#include <boost/lexical_cast.hpp>
#include "simp_simulation.h"
#include "global_gravity_simulation.h"
#include "multiply_simulation.h"
#include "test.h"
#include "mvp.h"
#include "pic.h"


int main(int argc, char **argv) {
    if (!strcmp(argv[1], "bench")){
        runBenchmark();
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
            multiplyRun(init_n, max_n, max_t, 0, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "4")){ //Huge
            multiplyRun(init_n, max_n, max_t, 1, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "5")){ //Static
            multiplyRun(init_n, max_n, max_t, 2, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "6")){ //Static advanced
            multiplyRun(init_n, max_n, max_t, 3, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "7")){ // Normal full
            multiplyRun(init_n, max_n, max_t, 4, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "8")){ // GPU Iterate with barrier using global memory
            multiplyRun(init_n, max_n, max_t, 5, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "9")){ // GPU Iterate with barrier using global memory with cooperative
            multiplyRun(init_n, max_n, max_t, 6, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "10")){ // GPU Iterate with barrier using global memory organised
            multiplyRun(init_n, max_n, max_t, 7, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "11")){ // GPU Iterate with barrier using multi block sync
            multiplyRun(init_n, max_n, max_t, 8, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "12")){ // Static GPU Full
            multiplyRun(init_n, max_n, max_t, 9, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "13")){ // Static GPU Full with cooperative
            multiplyRun(init_n, max_n, max_t, 10, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "14")){ // Dynamic with threads
            multiplyRun(init_n, max_n, max_t, 11, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "15")){ // Dynamic with blocks
            multiplyRun(init_n, max_n, max_t, 12, verbose, block_size, sleep_time, 0.01);
        }
        else if (!strcmp(argv[1], "16")){ // Dynamic with blocks and block checks
            multiplyRun(init_n, max_n, max_t, 13, verbose, block_size, sleep_time, 0.01);
        }
        else{
            float split_chance = boost::lexical_cast<float>(argv[8]);
            // MVP
            if (!strcmp(argv[1], "test")){ //Unit test
                runUnitTest(init_n, max_n, max_t, verbose, block_size, sleep_time, split_chance);
            }
            else if (!strcmp(argv[1], "20")){ // MVP Naive
                runMVP(init_n, max_n, max_t, 0, verbose, block_size, sleep_time, 0.01, split_chance);
            }
            else if (!strcmp(argv[1], "21")){ // MVP CPU Sync
                runMVP(init_n, max_n, max_t, 1, verbose, block_size, sleep_time, 0.01, split_chance);
            }
            else if (!strcmp(argv[1], "22")){ // MVP Static
                runMVP(init_n, max_n, max_t, 2, verbose, block_size, sleep_time, 0.01, split_chance);
            }
            else if (!strcmp(argv[1], "23")){ // MVP Dynamic
                runMVP(init_n, max_n, max_t, 3, verbose, block_size, sleep_time, 0.01, split_chance);
            }
            // PIC
            else if (!strcmp(argv[1], "23")){ // MVP Dynamic
                runMVP(init_n, max_n, max_t, 3, verbose, block_size, sleep_time, 0.01, split_chance);
            }
            else{
                float remove_chance = boost::lexical_cast<float>(argv[9]);
                int poisson_timestep = boost::lexical_cast<int>(argv[10]);
                // PIC
                if (!strcmp(argv[1], "30")){ // PIC GOOD
                    runPIC(init_n, max_n, max_t, poisson_timestep, 0, verbose, block_size, sleep_time, split_chance, remove_chance);
                }
            }
        }
    }
    double time = end_cpu_timer(start);
    printf("CPU time of program: %f ms\n", time);
}