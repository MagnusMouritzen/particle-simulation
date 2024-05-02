#include <stdio.h>
#include "electron.h"
#include "cell.h"

__global__ void resetGrid(cudaPitchedPtr d_grid, int3 grid_size);

__global__ void particlesToGrid(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size);

__global__ void updateGrid(cudaPitchedPtr d_grid, double electric_force_constant, int3 grid_size);

__global__ void gridToParticles(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size);