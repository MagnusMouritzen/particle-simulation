#include <cuda_runtime.h>
#include <stdio.h>
#include <boost/lexical_cast.hpp>
#include "simp_simulation.h"
#include "global_gravity_simulation.h"
#include "multiply_simulation.h"


int main(int argc, char **argv) {
    if (argc != 4) {
        printf("Wrong number of arguments. First is program to run, second is number of particles, third is verbose or not.");
        return 0;
    }
    int n = boost::lexical_cast<int>(argv[2]);
    bool verbose = boost::lexical_cast<int>(argv[3]);
    if (!strcmp(argv[1], "0")) {
        simpSimulationRun(n);
    }
    else if (!strcmp(argv[1], "1")) {
        globalGravityRun(n, 101, false, verbose);
    }
    else if (!strcmp(argv[1], "2")) {
        globalGravityRun(n, 101, true, verbose);
    }
    else if (!strcmp(argv[1], "3")){ //Normal
        multiplyRun(1, 10, 100, 0, verbose);
    }
    else if (!strcmp(argv[1], "4")){ //Huge
        multiplyRun(1, 10, 100, 1, verbose);
    }
    else if (!strcmp(argv[1], "5")){ //Static
        multiplyRun(1, 1000, 10000, 2, verbose);
    }
}