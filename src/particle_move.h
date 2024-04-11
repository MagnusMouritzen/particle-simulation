#include "electron.h"
#include "random.h"
#include "cell.h"

__device__ int updateParticle(Electron* electron, Electron* new_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* rand_state, int i, int t, float3 sim_size);

__global__ void setup_particles(Electron* d_electrons, curandState* d_rand_states, int init_n, float3 sim_size, int3 grid_size);