#include <cuda_runtime.h>
#include <stdio.h>
#include "simp_simulation.h"
#include "global_gravity_simulation.h"

int main(int argc, char **argv) {

    if (argc != 2) {
        return 0;
    }
    if (!strcmp(argv[1], "0")) {
        simpSimulationRun(1000);
    }
    else if (!strcmp(argv[1], "1")) {
        globalGravityRun(1000);
    }
}