#include "grid_operations.h"

#define getGridCell(x,y,z) (((Cell*)((((char*)d_grid.ptr) + z * (d_grid.pitch * grid_size.y)) + y * d_grid.pitch))[x])

__global__ void resetGrid(cudaPitchedPtr d_grid, int3 grid_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    getGridCell(x,y,z).charge = 0;
}

__global__ void particlesToGrid(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;

    Electron electron = d_electrons[i];

    int x = electron.position.x/cell_size;
    int y = electron.position.y/cell_size;
    int z = electron.position.z/cell_size;

    atomicAdd(&getGridCell(x,y,z).charge, 1);
}

__global__ void updateGrid(cudaPitchedPtr d_grid, double electric_force_constant, int3 grid_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // INSERT ELEMENT TO 3D ARRAY
    char* gridPtr = (char*)d_grid.ptr;
    size_t pitch = d_grid.pitch; // the number of bytes in a row of the array
    size_t slicePitch = pitch * grid_size.y; // the number of bytes pr slice
    char* slice = gridPtr + z * slicePitch; // get slice 
    char* row = (slice + y * pitch); // get row in slice

    double xAcc = 0;
    if (x != 0) xAcc -= ((Cell*)row)[x-1].charge;
    if (x != grid_size.x-1) xAcc += ((Cell*)row)[x+1].charge;
    xAcc *= electric_force_constant;

    double yAcc = 0;
    if (y != 0) yAcc -= ((Cell*)(row - pitch))[x].charge;
    if (y != grid_size.y-1) yAcc += ((Cell*)(row + pitch))[x].charge;
    yAcc *= electric_force_constant;

    double zAcc = 0;
    if (z != 0) zAcc -= ((Cell*)(row-slicePitch))[x].charge;
    if (z != grid_size.z-1) zAcc += ((Cell*)(row+slicePitch))[x].charge;
    zAcc *= electric_force_constant;

    ((Cell*)row)[x].acceleration = make_float3(xAcc, yAcc, zAcc);
}

__global__ void gridToParticles(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;

    Electron electron = d_electrons[i];

    int x = electron.position.x/cell_size;
    int y = electron.position.y/cell_size;
    int z = electron.position.z/cell_size;

    electron.acceleration =  getGridCell(x,y,z).acceleration;

    d_electrons[i] = electron;
}
