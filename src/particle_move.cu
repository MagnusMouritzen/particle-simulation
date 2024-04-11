#include "particle_move.h"
#include <stdio.h>

using namespace std;

__global__ void setup_particles(Electron* d_electrons, curandState* d_rand_states, int init_n, float3 sim_size, int3 grid_size) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;
    // d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], 0, sim_size.x), randFloat(&d_rand_states[i], 1, sim_size.y), randFloat(&d_rand_states[i], 1, sim_size.z));
    d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], (grid_size.x / 2 - 1)*cell_size, (grid_size.x / 2 + 2)*cell_size), 
                                          randFloat(&d_rand_states[i], (grid_size.y / 2 - 1)*cell_size, (grid_size.y / 2 + 2)*cell_size), 
                                          randFloat(&d_rand_states[i], (grid_size.z / 2 - 1)*cell_size, (grid_size.z / 2 + 2)*cell_size));
    d_electrons[i].weight = 1.0;
    d_electrons[i].timestamp = -1;
}

__device__ void leapfrog(Electron* electron, float delta_time){
    float delta_time_half = delta_time / 2;

    // Accelerate half
    electron->velocity.x -= electron->acceleration.x * delta_time_half;
    electron->velocity.y -= electron->acceleration.y * delta_time_half;
    electron->velocity.z -= electron->acceleration.z * delta_time_half;

    // Move
    electron->position.x += electron->velocity.x * delta_time;
    electron->position.y += electron->velocity.y * delta_time;
    electron->position.z += electron->velocity.z * delta_time;

    // Accelerate half
    electron->velocity.x -= electron->acceleration.x * delta_time_half;
    electron->velocity.y -= electron->acceleration.y * delta_time_half;
    electron->velocity.z -= electron->acceleration.z * delta_time_half;
}

__device__ bool checkOutOfBounds(Electron* electron, float3 sim_size){
    if (electron->position.x < 0
     || electron->position.x >= sim_size.x
     || electron->position.y < 0
     || electron->position.y >= sim_size.y
     || electron->position.z < 0
     || electron->position.z >= sim_size.z){
         electron->timestamp = DEAD;
         return true;
     }
     return false;
}

__device__ int collider(Electron* electron, Electron* new_electrons, float delta_time, int* n, int capacity, float split_chance, float remove_chance, curandState* rand_state, int i, int t){
    int new_i = -1;
    float rand = randFloat(rand_state, 0, 100);
    if (rand < split_chance) {
        if (*n < capacity) {
            new_i = atomicAdd(n, 1);
        
            if (new_i < capacity){
                Electron added_electron;
                added_electron = *electron;
                added_electron.creator = i;
                new_electrons[new_i] = added_electron;
            }
        }
        electron->velocity.x = -electron->velocity.x;
        electron->velocity.y = -electron->velocity.y;
        electron->velocity.z = -electron->velocity.z;
    }
    else if (rand < remove_chance + split_chance){
        electron->timestamp = DEAD;
        return new_i;
    }
    return new_i;
}

__device__ int updateParticle(Electron* electron, Electron* new_electrons, float delta_time, int* n, int capacity, float split_chance, float remove_chance, curandState* rand_state, int i, int t, float3 sim_size) {
    leapfrog(electron, delta_time);
    if (checkOutOfBounds(electron, sim_size)) return -1;
    return collider(electron, new_electrons, delta_time, n, capacity, split_chance, remove_chance, rand_state, i, t);
}