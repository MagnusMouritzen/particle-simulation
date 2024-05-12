#include "particle_move.h"
#include <stdio.h>

using namespace std;

__global__ void setup_particles(Electron* d_electrons, int init_n, float3 sim_size, int3 grid_size) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;

    Electron electron;
    newRandState(&electron.rand_state, i);

    electron.position = make_float3(randFloat(&electron.rand_state, 0, sim_size.x), randFloat(&electron.rand_state, 1, sim_size.y), randFloat(&electron.rand_state, 1, sim_size.z));
    electron.position = make_float3(randFloat(&electron.rand_state, (grid_size.x / 2 - 0)*cell_size, (grid_size.x / 2 + 2)*cell_size), 
                                          randFloat(&electron.rand_state, (grid_size.y / 2 - 0)*cell_size, (grid_size.y / 2 + 2)*cell_size), 
                                          randFloat(&electron.rand_state, (grid_size.z / 2 - 0)*cell_size, (grid_size.z / 2 + 2)*cell_size));

    // electron.position = make_float3(randFloat(&electron.rand_state, 0, (grid_size.x) * cell_size), 
    //                                       randFloat(&electron.rand_state, 0, (grid_size.y) * cell_size), 
    //                                       randFloat(&electron.rand_state, 0, (grid_size.z) * cell_size));
    // printf("x %d, y %d, z %d \n", (int)(electron.position.x/cell_size), (int)(electron.position.y/cell_size), (int)(electron.position.z/cell_size));
    electron.timestamp = -1;
    d_electrons[i] = electron;
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

__device__ bool collider(Electron* electron, Electron* new_electron, float delta_time, int i, int t, CSData* d_cross_sections){
    bool spawned_new = false;
    float rand = randFloat(&electron->rand_state, 0, 100);

    double electron_energy = (electron->velocity.x * electron->velocity.x) + 
                             (electron->velocity.y * electron->velocity.y) +
                             (electron->velocity.z * electron->velocity.z);
    int electron_energy_index = energyToIndex(electron_energy);

    // printf("x %d, y %d, z %d, energy : %e, index %d \n", (int)(electron->position.x/cell_size), (int)(electron->position.y/cell_size), (int)(electron->position.z/cell_size), electron_energy, electron_energy_index);
    
    float split_chance = d_cross_sections[electron_energy_index].split_chance;
    float remove_chance = d_cross_sections[electron_energy_index].remove_chance;
    
    if (rand < split_chance) {
        spawned_new = true;
        *new_electron = *electron;
        newRandState(&new_electron->rand_state, randInt(&electron->rand_state, 0, 2000000000)+100000000);
        new_electron->timestamp = t;

        electron->velocity.x = -electron->velocity.x;
        electron->velocity.y = -electron->velocity.y;
        electron->velocity.z = -electron->velocity.z;
    }
    else if (rand < remove_chance + split_chance){
        electron->timestamp = DEAD;
    }
    if (electron->position.x == 2.567073) printf("(%f, %f, %f)\n", electron->velocity.x, electron->velocity.y, electron->velocity.z);
    return spawned_new;
}

__device__ bool updateParticle(Electron* electron, Electron* new_electron, float delta_time, int i, int t, float3 sim_size, CSData* d_cross_sections) {
    leapfrog(electron, delta_time);
    if (checkOutOfBounds(electron, sim_size)) return false;
    return collider(electron, new_electron, delta_time, i, t, d_cross_sections);
}