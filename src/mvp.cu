#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "utility.h"
#include "mvp.h"


__device__ static void simulate(Electron* electrons, float deltaTime, int* n, int capacity, int i, int t){

    float myrandf = curand_uniform(rand_state+i);
    myrandf *= (10 - 5 +0.999999);
    myrandf += 5;
    int mob_steps = (int)truncf(myrandf);

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
                electrons[new_i].weight = electrons[i].weight;
                __threadfence();
                electrons[new_i].timestamp = t;
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
// Kernel for random numbers
__global__ void setup_kernel(curandState *state) {
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

__global__ static void updateNormalFull(Electron* d_electrons, float deltaTime, int* n, int start_n, int offset, int capacity, int max_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    for(int t = max(1, d_electrons[i].timestamp + 1); t <= max_t; t++){
        simulate(d_electrons, deltaTime, n, capacity, i, t);
    }
}

__global__ static void updateGPUIterate(Electron* electrons, float deltaTime, int* n, int capacity, int max_t, int* wait_counter, int sleep_time_ns, int* n_done) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;


    for (int i = thread_id; i < capacity; i += num_blocks * block_size) {

        while(electrons[i].timestamp == 0) {
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        for(int t=max(1,electrons[i].timestamp+1); t<=max_t; t++) { //update particle from next time iteration
            simulate(electrons, deltaTime, n, capacity, i, t);
        }
        // __threadfence(); // is it needed here?
        atomicAdd(n_done,1);
    }

}

__global__ static void updateDynamicBlocks(Electron* electrons, float deltaTime, int* n, int capacity, int max_t, int* wait_counter, int sleep_time_ns, int* n_done, int* i_global, int* i_blocks) {

    while (true) {
        __syncthreads(); //sync threads seems to be able to handle threads being terminated
        if (threadIdx.x==0) {
            i_blocks[blockIdx.x] = atomicAdd(i_global, blockDim.x);
        }
        __syncthreads();

        int i = i_blocks[blockIdx.x] + threadIdx.x;

        if (i >= capacity) break;

        while (electrons[i].timestamp == 0) {
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        for (int t=max(1,electrons[i].timestamp+1); t<=max_t; t++) { //update particle from next time iteration
            simulate(electrons, deltaTime, n, capacity, i, t);
        }

        // __threadfence(); // is it needed here?
        atomicAdd(n_done,1);

    }
}


void runMVP (int init_n, int capacity, int max_t, int mode, int verbose, int block_size, int sleep_time_ns, float delta_time) {


    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));

    setup_kernel<<<10, block_size>>>(d_state);
    
    Electron* h_electrons = (Electron *)calloc(capacity, sizeof(Electron));
    for(int i=0; i<init_n; i++) {
        h_electrons[i].position = make_float3(250, 250, 1.0);
        h_electrons[i].weight = 1.0;
        h_electrons[i].timestamp = -1;
    }

    Electron* d_electrons;
    cudaMalloc(&d_electrons, capacity * sizeof(Electron));

    cudaMemcpy(d_electrons, h_electrons, init_n * sizeof(Electron), cudaMemcpyHostToDevice);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);

    int* waitCounter;
    cudaMalloc(&waitCounter, 2 * sizeof(int));
    cudaMemset(waitCounter, 0, 2 * sizeof(int));

    int* n_done;
    cudaMalloc(&n_done, sizeof(int));
    cudaMemset(n_done, 0, sizeof(int));

    int* i_global;
    cudaMalloc(&i_global, sizeof(int));
    cudaMemset(i_global, 0, sizeof(int));

    switch(mode){
        case 0: { //Naive

            break;
        }
        case 1: { //CPU Sync
            int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
            while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                updateNormalFull<<<num_blocks, block_size>>>(electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, max_t);
                last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
            }
            
            break;
        }
        case 2: { //Static
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);

            updateGPUIterate<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done);
            
            break;
        }
        case 3: { //Dynamic
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);

            int* i_blocks;
            cudaMalloc(&i_blocks, num_blocks*sizeof(int));
            cudaMemset(i_blocks, 0, num_blocks*sizeof(int));

            updateDynamicBlocks<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done, i_global, i_blocks);

            break;
        }

        default:
            break;
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s \n", cudaGetErrorString(error));
        // Handle error appropriately
    }


    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_electrons, d_electrons, min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);   
    cudaEventSynchronize(stop);

    free(n_host);
    cudaFree(d_electrons);
    cudaFree(n);
    cudaFree(n_done);
    cudaFree(waitCounter);

}