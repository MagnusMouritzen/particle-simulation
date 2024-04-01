#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include "mvp.h"

#define DEAD -2

// __device__ curandState *d_rand_state;

__shared__ int i_block;
__shared__ int capacity;

__shared__ int n_block;
__shared__ int new_i_block;

__device__ static void newRandState(curandState* d_rand_states, int i, int seed){
    curand_init(1234, seed, 0, &d_rand_states[i]);
}

__device__ static float randFloat(curandState* state, float min, float max){
    float rand = curand_uniform(state);
    rand *= (max - min + 0.999999);
    rand += min;
    return rand;
}

__device__ static int randInt(curandState* state, int min, int max){
    float rand_float = randFloat(state, min, max);
    return (int)truncf(rand_float);
}

// Kernel for random numbers
__global__ static void setup(Electron* d_electrons, curandState* d_rand_states, int init_n) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;
    newRandState(d_rand_states, i, i);
    d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], 1, 499), randFloat(&d_rand_states[i], 1, 499), 1.0);
    d_electrons[i].weight = 1.0;
    d_electrons[i].timestamp = -1;
}

__device__ static int updateParticle(Electron* electron, Electron* new_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* rand_state, int i, int t){
    electron->velocity.y -= 9.82 * deltaTime * electron->weight;
    electron->position.y += electron->velocity.y * deltaTime;

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
                added_electron.position.x = electron->position.x + added_electron.velocity.x * deltaTime;
                added_electron.position.z = 1.0;
                added_electron.velocity.z = 1.0;
                added_electron.weight = electron->weight;
                added_electron.creator = i;
                
                new_electrons[new_i] = added_electron;
            }
        }
    }
    else if (rand - split_chance < remove_chance){
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


__device__ static void simulateMany(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* d_rand_states, int i, int start_t, int poisson_timestep){
    Electron electron = d_electrons[i];
    curandState rand_state = d_rand_states[i];

    for(int t = start_t; t <= poisson_timestep; t++){
        int new_i = updateParticle(&electron, d_electrons, deltaTime, n, capacity, split_chance, remove_chance, &rand_state, i, t);
        if(new_i != -1 && new_i < capacity) {  // If a new particle was spawned and there is room for it.
            newRandState(d_rand_states, new_i, randInt(&rand_state, 0, 10000));
            __threadfence();
            d_electrons[new_i].timestamp = t;
            printf("%d: (%d) NEW %d {%f}", i, t, new_i, d_electrons[new_i].position.x);
        }
        else if (electron.timestamp == DEAD){  // If particle is to be removed.
            printf("%d: (%d) DEAD\n", i, t);
            break;
        }
    }
    if (electron.timestamp != DEAD) electron.timestamp = -1;

    d_electrons[i] = electron;
    d_rand_states[i] = rand_state;
}

__global__ static void poisson(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_done, int* i_global) {

    while (true) {
        __syncthreads(); //sync threads seems to be able to handle threads being terminated
        if (threadIdx.x==0) {
            i_block = atomicAdd(i_global, blockDim.x);
        }
        __syncthreads();

        int i = i_block + threadIdx.x;

        if (i >= capacity) break;

        while (d_electrons[i].timestamp == 0) {
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        simulateMany(d_electrons, deltaTime, n, capacity, split_chance, remove_chance, d_rand_states, i, max(1, d_electrons[i].timestamp + 1), poisson_timestep);
        atomicAdd(n_done,1);

    }
}

__global__ static void remove_dead_particles(Electron* d_electrons_old, Electron* d_electrons_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    int i_local = -1;
    if (d_electrons_old[i].timestamp != DEAD){
        i_local = atomicAdd(&n_block, 1);
    }
    printf("%d: n %d, start n %d, i local %d\n", i, *n, start_n, i_local);

    __syncthreads();
    if (threadIdx.x == 0){
        i_block = atomicAdd(n, n_block);
    }
    __syncthreads();
    
    printf("%d: i block %d\n", i, i_block);

    if (i_local == -1) return;
    d_electrons_new[i_block + i_local] = d_electrons_old[i];
}

static void log(int verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity){
    if (verbose == 0 || t % verbose != 0) return;
    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    int true_n = min(*n_host, capacity);
    cudaMemcpy(electrons_host, electrons, true_n * sizeof(Electron), cudaMemcpyDeviceToHost);
    printf("Time %d, amount %d\n", t, *n_host);
    for(int i = 0; i < true_n; i++){
        electrons_host[i].print(i);
    }
    image(true_n, electrons_host, t); // visualize a snapshot of the current positions of the particles     
    printf("\n");
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s \n", cudaGetErrorString(error));
        throw runtime_error(cudaGetErrorString(error));
        // Handle error appropriately
    }
}

RunData runPIC (int init_n, int capacity, int poisson_steps, int poisson_timestep, int mode, int verbose, int block_size, int sleep_time_ns, float split_chance, float remove_chance) {
    printf("MVP with\ninit n: %d\ncapacity: %d\npoisson steps: %d\npoisson_timestep: %d\nblock size: %d\nsleep time: %d\nsplit chance: %f\nremove chance: %f\n", init_n, capacity, poisson_steps, poisson_timestep, block_size, sleep_time_ns, split_chance, remove_chance);

    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = poisson_steps;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;
    timing_data.split_chance = split_chance;
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Electron* h_electrons = (Electron *)calloc(capacity, sizeof(Electron));
    Electron* d_electrons;
    cudaMalloc(&d_electrons, 2 * capacity * sizeof(Electron));
    cudaMemset(d_electrons, 0, 2 * capacity * sizeof(Electron));

    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, capacity * sizeof(curandState));
    setup<<<(init_n + block_size - 1) / block_size, block_size>>>(d_electrons, d_rand_states, init_n);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);

    int* n_done;
    cudaMalloc(&n_done, sizeof(int));
    
    int* i_global;
    cudaMalloc(&i_global, sizeof(int));


    switch(mode){
        case 0: { // GOOD
            timing_data.function = "GOOD";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            cudaEventRecord(start);

            int source_index = 0;
            int destination_index = 0;
            for (int t = 0; t < poisson_steps; t++)
            {
                printf("New time step %d\n", t);
                source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
                destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.
                printf("%d: n %d, source %d, dest %d\n", t, *n_host, source_index, destination_index);

                log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
                cudaMemset(n_done, 0, sizeof(int));
                cudaMemset(i_global, 0, sizeof(int));
                printf("Poisson\n");
                poisson<<<num_blocks, block_size>>>(&d_electrons[source_index], 0.1, n, capacity, split_chance, remove_chance, d_rand_states, poisson_timestep, sleep_time_ns, n_done, i_global);
                log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemset(n, 0, sizeof(int));
                int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                printf("Remove %d\n", num_blocks_all);
                remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
            }
            log(verbose, poisson_steps, h_electrons, &d_electrons[destination_index], n_host, n, capacity);
            
            
            cudaEventRecord(stop);
            break;
        }
        default:
            break;
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s \n", cudaGetErrorString(error));
        throw runtime_error(cudaGetErrorString(error));
        // Handle error appropriately
    }



    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_electrons, d_electrons, min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);   

    cudaEventSynchronize(stop);
    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("GPU time of program: %f ms\n", runtime_ms);
    timing_data.time = runtime_ms;
    timing_data.final_n = min(*n_host, capacity);

    RunData run_data;
    run_data.timing_data = timing_data;
    run_data.final_n = min(*n_host, capacity);
    run_data.electrons = new Electron[capacity];
    memcpy(run_data.electrons, h_electrons, capacity * sizeof(Electron));

    free(n_host);
    free(h_electrons);
    cudaFree(d_electrons);
    cudaFree(n);
    cudaFree(n_done);
    cudaFree(i_global);
    cudaFree(d_rand_states);

    return run_data;
}