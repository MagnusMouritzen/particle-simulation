#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "utility.h"
#include "pic_simulation.h"

#define MIN 2
#define MAX 7
#define ITER 10000000

__device__ static void simulate(Electron* electrons, float deltaTime, int* n, int capacity, int i, int t){
    electrons[i].velocity.y -= 9.82 * deltaTime * electrons[i].weight;
    electrons[i].position.y += electrons[i].velocity.y * deltaTime;

    if (electrons[i].position.y <= 0){
        electrons[i].position.y = -electrons[i].position.y;
        electrons[i].velocity.y = -electrons[i].velocity.y;

        if (*n < capacity) {
            int new_i = atomicAdd(n, 1);
        
            if (new_i < capacity){
                if (electrons[i].velocity.x >= 0){
                    electrons[i].velocity.x += 10;
                }
                else{
                    electrons[i].velocity.x -= 10;
                }

                //printf("Particle %d spawns particle %d\n", i, new_i);
                electrons[new_i].position.y = electrons[i].position.y;
                electrons[new_i].velocity.y = electrons[i].velocity.y;
                if (electrons[i].velocity.x >= 0){
                    electrons[new_i].velocity.x = electrons[i].velocity.x - 20;
                }
                else{
                    electrons[new_i].velocity.x = electrons[i].velocity.x + 20;
                }
                electrons[new_i].position.x = electrons[i].position.x + electrons[new_i].velocity.x * deltaTime;
                electrons[new_i].timestamp = t;
                electrons[new_i].weight = electrons[i].weight;
            }
        }
    }
    else if (electrons[i].position.y >= 500){
        electrons[i].position.y = 500 - (electrons[i].position.y - 500);
        electrons[i].velocity.y = -electrons[i].velocity.y;
    }

    electrons[i].position.x += electrons[i].velocity.x * deltaTime;

    if (electrons[i].position.x <= 0){
        electrons[i].position.x = -electrons[i].position.x;
        electrons[i].velocity.x = -electrons[i].velocity.x;
        electrons[i].weight *= -1;
    }
    else if (electrons[i].position.x >= 500){
        electrons[i].position.x = 500 - (electrons[i].position.x - 500);
        electrons[i].velocity.x = -electrons[i].velocity.x;
        electrons[i].weight *= -1;
    }
}

__global__ void setup_kernel(curandState *state) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ static void updateNormalFull(Electron* electrons, float deltaTime, int* n, int start_n, int offset, int capacity, curandState *rand_state) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    float myrandf = curand_uniform(rand_state+i);
    myrandf *= (10 - 5 +0.999999);
    myrandf += 5;
    int mob_steps = (int)truncf(myrandf);

    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    for(int t = max(1, electrons[i].timestamp + 1); t <= mob_steps; t++){
        simulate(electrons, deltaTime, n, capacity, i, t);
    }
}

// static void log(int verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity){
//     if (verbose == 0 || t % verbose != 0) return;
//     cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
//     int true_n = min(*n_host, capacity);
//     cudaMemcpy(electrons_host, electrons, true_n * sizeof(Electron), cudaMemcpyDeviceToHost);
//     printf("Time %d, amount %d\n", t, *n_host);
//     for(int i = 0; i < true_n; i++){
//         printf("%d: (%.6f, %.6f) (%.6f, %.6f)\n", i, electrons_host[i].position.x, electrons_host[i].position.y, electrons_host[i].velocity.x, electrons_host[i].velocity.y);
//     }
//     image(true_n, electrons_host, t); // visualize a snapshot of the current positions of the particles     
//     printf("\n");
// }

void runPIC(int init_n, int capacity, int max_t, int verbose, int block_size) {

    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));

    setup_kernel<<<10,block_size>>>(d_state);



    printf("PIC with\ninit n: %d\ncapacity: %d\nmax t: %d\nblock size: %d\n", init_n, capacity, max_t, block_size);
    
    Electron* electrons_host = (Electron *)calloc(capacity, sizeof(Electron));
    for(int i=0; i<init_n; i++) {
        electrons_host[i].position = make_float3(250, 250, 1.0);
        electrons_host[i].weight = 1.0;
        electrons_host[i].timestamp = -1;
    }

    float delta_time = 0.1;

    Electron* electrons;
    cudaMalloc(&electrons, capacity * sizeof(Electron));

    cudaMemcpy(electrons, electrons_host, capacity * sizeof(Electron), cudaMemcpyHostToDevice);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);
    
    if (verbose) printf("Time %d, amount %d\n", 0, *n_host);


    printf("PIC: normal full\n");
    for(int i = 0; i < max_t; i++) { // Poisson

        int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
        while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
            int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
            updateNormalFull<<<num_blocks, block_size>>>(electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, d_state);
            last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
            cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
        }

    }

    // log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);
    
    
    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(electrons_host, electrons, min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);

}