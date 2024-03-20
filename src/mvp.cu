#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include "mvp.h"

// __device__ curandState *d_rand_state;

__shared__ int i_block;
__shared__ int capacity;

__shared__ int n_block;
__shared__ int new_i_block;

__device__ void newRandState(curandState* d_rand_states, int i, int seed){
    // printf("New rand state for %d seed %d\n", i, seed);
    curand_init(1234, seed, 0, &d_rand_states[i]);
}

__device__ float randFloat(curandState* state, float min, float max){
    float rand = curand_uniform(state);
    rand *= (max - min + 0.999999);
    rand += min;
    return rand;
}

__device__ int randInt(curandState* state, int min, int max){
    float rand_float = randFloat(state, min, max);
    return (int)truncf(rand_float);
}

// Kernel for random numbers
__global__ void setup(Electron* d_electrons, curandState* d_rand_states, int init_n) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    if (i >= init_n) return;
    newRandState(d_rand_states, i, i);
    d_electrons[i].position = make_float3(randFloat(&d_rand_states[i], 1, 499), randFloat(&d_rand_states[i], 1, 499), 1.0);
    d_electrons[i].weight = 1.0;
    d_electrons[i].timestamp = -1;
}

__device__ static int updateParticle(Electron* electron, Electron* new_electrons, float deltaTime, int* n, int capacity, float split_chance, curandState* d_rand_states, int i, int t){
    electron->velocity.y -= 9.82 * deltaTime * electron->weight;
    electron->position.y += electron->velocity.y * deltaTime;

    int new_i = -1;
    float rand = randFloat(&d_rand_states[i], 0, 100);
    // printf("%d: %.02f\n", i, rand);
    if (rand < split_chance) {
        if (*n < capacity) {
            // printf("n %d\n", *n);
            new_i = atomicAdd(n, 1);
        
            if (new_i < capacity){

                if (electron->velocity.x >= 0){
                    electron->velocity.x += 10;
                }
                else{
                    electron->velocity.x -= 10;
                }

                
                new_electrons[new_i].position.y = electron->position.y;
                new_electrons[new_i].velocity.y = electron->velocity.y;
                if (electron->velocity.x >= 0){
                    new_electrons[new_i].velocity.x = electron->velocity.x - 20;
                }
                else{
                    new_electrons[new_i].velocity.x = electron->velocity.x + 20;
                }
                new_electrons[new_i].position.x = electron->position.x + new_electrons[new_i].velocity.x * deltaTime;
                new_electrons[new_i].position.z = 1.0;
                new_electrons[new_i].velocity.z = 1.0;
                new_electrons[new_i].weight = electron->weight;
                new_electrons[new_i].creator = i;
                __threadfence();
                new_electrons[new_i].timestamp = t;
            }
        }
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

__device__ static void simulateNaive(Electron* d_electrons, Electron* new_electrons, float deltaTime, int* n, int capacity, float split_chance, curandState* d_rand_states, int i, int t){
    updateParticle(&d_electrons[i], new_electrons, deltaTime, n, capacity, split_chance, d_rand_states, i, t);
}

__device__ static void simulateMany(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, curandState* d_rand_states, int i, int start_t, int max_t){
    Electron electron = d_electrons[i];

    for(int t = start_t; t <= max_t; t++){
        int new_i = updateParticle(&electron, d_electrons, deltaTime, n, capacity, split_chance, d_rand_states, i, t);
        if(new_i != -1) {
            printf("Particle %d spawns particle %d\n", i, new_i);
            newRandState(d_rand_states, new_i, randInt(&d_rand_states[i], 0, 10000));
        }
    }
    d_electrons[i] = electron;
}

__global__ static void naive(Electron* d_electrons, float deltaTime, int* n, int start_n, int capacity, float split_chance, curandState* d_rand_states, int t) {
    //__shared__ char sharedMemory[sizeof(Electron) * 1024]; // Allocate raw shared memory
    //Electron* new_particles_block = reinterpret_cast<Electron*>(sharedMemory);
    __shared__ Electron new_particles_block[1024];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) n_block = 0;

    __syncthreads(); // Ensure construction is finished
    
    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    simulateNaive(d_electrons, new_particles_block, deltaTime, &n_block, capacity, split_chance, d_rand_states, i, t);

    __syncthreads();

    if (threadIdx.x == 0){
        if (*n < capacity) new_i_block = atomicAdd(n, n_block);  // Avoid risk of n overflowing int max value
        else new_i_block = capacity;
    }

    __syncthreads();

    if (threadIdx.x >= n_block) return;
    int global_i = new_i_block + threadIdx.x;
    if (global_i >= capacity) return;
    newRandState(d_rand_states, global_i, randInt(&d_rand_states[new_particles_block[threadIdx.x].creator], 0, 10000));
    d_electrons[global_i] = new_particles_block[threadIdx.x];
}

__global__ static void cpuSynch(Electron* d_electrons, float deltaTime, int* n, int start_n, int offset, int capacity, float split_chance, curandState* d_rand_states, int max_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    simulateMany(d_electrons, deltaTime, n, capacity, split_chance, d_rand_states, i, max(1, d_electrons[i].timestamp + 1), max_t);
}

__global__ static void staticGpu(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, curandState* d_rand_states, int max_t, int sleep_time_ns, int* n_done) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;
    int n_done_local = 0;

    for (int i = thread_id; i < capacity; i += num_blocks * block_size) {

        while(d_electrons[i].timestamp == 0) {
            if (n_done_local != 0){
                atomicAdd(n_done, n_done_local);
                n_done_local = 0;
            }
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        simulateMany(d_electrons, deltaTime, n, capacity, split_chance, d_rand_states, i, max(1, d_electrons[i].timestamp + 1), max_t);
        n_done_local++;
    }

    if (n_done_local != 0){
        atomicAdd(n_done, n_done_local);
    }

}

__global__ static void dynamicGpu(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, curandState* d_rand_states, int max_t, int sleep_time_ns, int* n_done, int* i_global) {

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

        simulateMany(d_electrons, deltaTime, n, capacity, split_chance, d_rand_states, i, max(1, d_electrons[i].timestamp + 1), max_t);
        atomicAdd(n_done,1);

    }
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

RunData runMVP (int init_n, int capacity, int max_t, int mode, int verbose, int block_size, int sleep_time_ns, float delta_time, float split_chance) {
    printf("MVP with\ninit n: %d\ncapacity: %d\nmax t: %d\nblock size: %d\nsleep time: %d\ndelta time: %f\n", init_n, capacity, max_t, block_size, sleep_time_ns, delta_time);

    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = max_t;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Electron* h_electrons = (Electron *)calloc(capacity, sizeof(Electron));
    Electron* d_electrons;
    cudaMalloc(&d_electrons, capacity * sizeof(Electron));
    cudaMemset(d_electrons, 0, capacity * sizeof(Electron));

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
    cudaMemset(n_done, 0, sizeof(int));
    
    int* i_global;
    cudaMalloc(&i_global, sizeof(int));
    cudaMemset(i_global, 0, sizeof(int));

    switch(mode){
        case 0: { //Naive      
            timing_data.function = "Naive";
            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++){
                int num_blocks = (min(*n_host, capacity) + block_size - 1) / block_size;
                naive<<<num_blocks, block_size>>>(d_electrons, delta_time, n, min(*n_host, capacity), capacity, split_chance, d_rand_states, t);
                log(verbose, t, h_electrons, d_electrons, n_host, n, capacity);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
            }
            cudaEventRecord(stop);
            break;
        }
        case 1: { //CPU Sync
            timing_data.function = "CPU Sync";
            cudaEventRecord(start);
            int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
            while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                cpuSynch<<<num_blocks, block_size>>>(d_electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, split_chance, d_rand_states, max_t);
                last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
            }
            cudaEventRecord(stop);
            break;
        }
        case 2: { //Static
            timing_data.function = "Static";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            cudaEventRecord(start);
            staticGpu<<<num_blocks, block_size>>>(d_electrons, delta_time, n, capacity, split_chance, d_rand_states, max_t, sleep_time_ns, n_done);
            cudaEventRecord(stop);
            
            break;
        }
        case 3: { //Dynamic
            timing_data.function = "Dynamic";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            cudaEventRecord(start);
            dynamicGpu<<<num_blocks, block_size>>>(d_electrons, delta_time, n, capacity, split_chance, d_rand_states, max_t, sleep_time_ns, n_done, i_global);
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

    log(verbose, max_t, h_electrons, d_electrons, n_host, n, capacity);



    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_electrons, d_electrons, min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);   

    cudaEventSynchronize(stop);
    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("GPU time of program: %f ms\n", runtime_ms);
    timing_data.time = runtime_ms;

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