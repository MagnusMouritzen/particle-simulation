#include "electron.h"
#include "random.h"
#include "cell.h"
#include "cross_section.h"

__device__ bool updateParticle(Electron* electron, Electron* new_electron, float delta_time, curandState* rand_state, int i, int t, float3 sim_size, CSData* d_cross_sections);

__global__ void setup_particles(Electron* d_electrons, curandState* d_rand_states, int init_n, float3 sim_size, int3 grid_size);