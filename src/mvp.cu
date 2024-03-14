#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstring>
#include <math.h>
#include "mvp.h"

// __device__ curandState *d_rand_state;

__shared__ int i_block;
__shared__ int capacity;

__device__ static void simulate(Electron* d_electrons, float deltaTime, int* n, int capacity, int i, int t){

    // float myrandf = curand_uniform(d_rand_state+i);
    // float min = 5;
    // float max = 10;
    // myrandf *= (max - min +0.999999);
    // myrandf += min;

    // int mob_steps = (int)truncf(myrandf);

    // printf("random %d", mob_steps);

    d_electrons[i].velocity.y -= 9.82 * deltaTime * d_electrons[i].weight;
    d_electrons[i].position.y += d_electrons[i].velocity.y * deltaTime;

    if (d_electrons[i].position.y <= 0){
        d_electrons[i].position.y = -d_electrons[i].position.y;
        d_electrons[i].velocity.y = -d_electrons[i].velocity.y;

        if (*n < capacity) {
            int new_i = atomicAdd(n, 1);
        
            if (new_i < capacity){
                if (d_electrons[i].velocity.x >= 0){
                    d_electrons[i].velocity.x += 10;
                }
                else{
                    d_electrons[i].velocity.x -= 10;
                }

                //printf("Particle %d spawns particle %d\n", i, new_i);
                d_electrons[new_i].position.y = d_electrons[i].position.y;
                d_electrons[new_i].velocity.y = d_electrons[i].velocity.y;
                if (d_electrons[i].velocity.x >= 0){
                    d_electrons[new_i].velocity.x = d_electrons[i].velocity.x - 20;
                }
                else{
                    d_electrons[new_i].velocity.x = d_electrons[i].velocity.x + 20;
                }
                d_electrons[new_i].position.x = d_electrons[i].position.x + d_electrons[new_i].velocity.x * deltaTime;
                d_electrons[new_i].weight = d_electrons[i].weight;
                __threadfence();
                d_electrons[new_i].timestamp = t;
            }
        }
    }
    else if (d_electrons[i].position.y >= 500){
        d_electrons[i].position.y = 500 - (d_electrons[i].position.y - 500);
        d_electrons[i].velocity.y = -d_electrons[i].velocity.y;
    }

    d_electrons[i].position.x += d_electrons[i].velocity.x * deltaTime;

    if (d_electrons[i].position.x <= 0){
        d_electrons[i].position.x = -d_electrons[i].position.x;
        d_electrons[i].velocity.x = -d_electrons[i].velocity.x;
        d_electrons[i].weight *= -1;
    }
    else if (d_electrons[i].position.x >= 500){
        d_electrons[i].position.x = 500 - (d_electrons[i].position.x - 500);
        d_electrons[i].velocity.x = -d_electrons[i].velocity.x;
        d_electrons[i].weight *= -1;
    }
}
// // Kernel for random numbers
// __global__ void setup_kernel() {
//     int idx = threadIdx.x+blockDim.x*blockIdx.x;
//     curand_init(1234, idx, 0, &d_rand_state[idx]);
// }

__global__ static void naive(Electron* d_electrons, float deltaTime, int* n, int start_n, int capacity, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    simulate(d_electrons, deltaTime, n, capacity, i, t);
}

__global__ static void cpuSynch(Electron* d_electrons, float deltaTime, int* n, int start_n, int offset, int capacity, int max_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    for(int t = max(1, d_electrons[i].timestamp + 1); t <= max_t; t++){
        simulate(d_electrons, deltaTime, n, capacity, i, t);
    }
}

__global__ static void staticGpu(Electron* d_electrons, float deltaTime, int* n, int capacity, int max_t, int sleep_time_ns, int* n_done) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;


    for (int i = thread_id; i < capacity; i += num_blocks * block_size) {

        while(d_electrons[i].timestamp == 0) {
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        for(int t=max(1,d_electrons[i].timestamp+1); t<=max_t; t++) { //update particle from next time iteration
            simulate(d_electrons, deltaTime, n, capacity, i, t);
        }
        // __threadfence(); // is it needed here?
        atomicAdd(n_done,1);
    }

}

__global__ static void dynamicGpu(Electron* d_electrons, float deltaTime, int* n, int capacity, int max_t, int sleep_time_ns, int* n_done, int* i_global) {

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

        for (int t=max(1,d_electrons[i].timestamp+1); t<=max_t; t++) { //update particle from next time iteration
            simulate(d_electrons, deltaTime, n, capacity, i, t);
        }

        // __threadfence(); // is it needed here?
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
}

RunData runMVP (int init_n, int capacity, int max_t, int mode, int verbose, int block_size, int sleep_time_ns, float delta_time) {

    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = max_t;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // cudaMalloc(&d_rand_state, sizeof(curandState));

    // setup_kernel<<<1, 1>>>();
    
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
                naive<<<num_blocks, block_size>>>(d_electrons, delta_time, n, min(*n_host, capacity), capacity, t);

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
                cpuSynch<<<num_blocks, block_size>>>(d_electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, max_t);
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
            staticGpu<<<num_blocks, block_size>>>(d_electrons, delta_time, n, capacity, max_t, sleep_time_ns, n_done);
            cudaEventRecord(stop);
            
            break;
        }
        case 3: { //Dynamic
            timing_data.function = "Dynamic";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            cudaEventRecord(start);
            dynamicGpu<<<num_blocks, block_size>>>(d_electrons, delta_time, n, capacity, max_t, sleep_time_ns, n_done, i_global);
            cudaEventRecord(stop);
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

    return run_data;
}