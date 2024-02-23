#include <cuda_runtime.h>
#include <stdio.h>
#include <boost/lexical_cast.hpp>
#include "simp_simulation.h"
#include "global_gravity_simulation.h"
#include "multiply_simulation.h"
#include "utility.h"


int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Too few arguments.\n1 - Mode.\n2 - Verbose.\n3 - Initial particle amount.\n4 - Max iterations.\n5 - Max particles.\n");
        return 0;
    }
    bool verbose = boost::lexical_cast<int>(argv[2]);
    int init_n = boost::lexical_cast<int>(argv[3]);
    int max_t = boost::lexical_cast<int>(argv[4]);
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
        int max_n = boost::lexical_cast<int>(argv[5]);
        if (!strcmp(argv[1], "3")){ //Normal
            multiplyRun(init_n, max_n, max_t, 0, verbose);
        }
        else if (!strcmp(argv[1], "4")){ //Huge
            multiplyRun(init_n, max_n, max_t, 1, verbose);
        }
        else if (!strcmp(argv[1], "5")){ //Static
            multiplyRun(init_n, max_n, max_t, 2, verbose);
        }
    }
    double time = end_cpu_timer(start);
    printf("CPU time of program: %f ms\n", time);
}