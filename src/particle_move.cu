#include "particle_move.h"
#include <stdio.h>

using namespace std;

__global__ void setup_particles(Electron* d_electrons, curandState* d_rand_states, int init_n, float3 sim_size) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;
    d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], 0, sim_size.x), randFloat(&d_rand_states[i], 1, sim_size.y), randFloat(&d_rand_states[i], 1, sim_size.z));
    d_electrons[i].weight = 1.0;
    d_electrons[i].timestamp = -1;
}

__device__ int updateParticle(Electron* electron, Electron* new_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* rand_state, int i, int t) {
    electron->velocity.y -= 9.82 * deltaTime * electron->weight;
    electron->position.y += electron->velocity.y * deltaTime;

    electron->position.z += electron->velocity.z * deltaTime;

    int new_i = -1;
    float rand = randFloat(rand_state, 0, 100);
    printf("%d: (%d) rand %f\n", i, t, rand);
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
                added_electron.position.x = electron->position.x + added_electron.velocity.x * deltaTime;
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

    if (electron->position.y <= 0){
        electron->position.y = -electron->position.y;
        electron->velocity.y = -electron->velocity.y;
    }
    else if (electron->position.y >= 500){
        electron->position.y = 500 - (electron->position.y - 500);
        electron->velocity.y = -electron->velocity.y;
    }

    electron->position.x += electron->velocity.x * deltaTime;

    if (electron->position.x <= 0){
        electron->position.x = -electron->position.x;
        electron->velocity.x = -electron->velocity.x;
        electron->weight *= -1;
    }
    else if (electron->position.x >= 500){
        electron->position.x = 500 - (electron->position.x - 500);
        electron->velocity.x = -electron->velocity.x;
        electron->weight *= -1;
    }
    return new_i;
}