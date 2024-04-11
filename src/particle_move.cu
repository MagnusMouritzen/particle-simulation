#include "particle_move.h"
#include <stdio.h>

using namespace std;

__global__ void setup_particles(Electron* d_electrons, curandState* d_rand_states, int init_n) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;
    d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], 1, 499), randFloat(&d_rand_states[i], 1, 499), randFloat(&d_rand_states[i], 1, 499));
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

                if (electron->velocity.x >= 0){
                    electron->velocity.x += 10;
                }
                else{
                    electron->velocity.x -= 10;
                }

                Electron added_electron;

                added_electron.position.y = electron->position.y;
                added_electron.velocity.y = electron->velocity.y;
                if (electron->velocity.x >= 0){
                    added_electron.velocity.x = electron->velocity.x - 20;
                }
                else{
                    added_electron.velocity.x = electron->velocity.x + 20;
                }
                added_electron.position.x = electron->position.x + added_electron.velocity.x * delta_time_half;
                added_electron.position.z = electron->position.z;
                added_electron.velocity.z = electron->velocity.z;
                added_electron.weight = electron->weight;
                added_electron.creator = i;
                
                new_electrons[new_i] = added_electron;
            }
        }
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