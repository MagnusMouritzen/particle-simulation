#include <cuda_runtime.h>
#include <stdio.h>
#include "utility.h"
#include "multiply_simulation.h"

__device__ static void simulate(Electron* electrons, float deltaTime, int* n, int capacity, int i, int t){
    electrons[i].velocity.y -= 9.82 * deltaTime;
    electrons[i].position.y += electrons[i].velocity.y * deltaTime;

    if (electrons[i].position.y <= 0){
        electrons[i].position.y = -electrons[i].position.y;
        electrons[i].velocity.y = -electrons[i].velocity.y;
        if (electrons[i].velocity.x == 0){
            electrons[i].velocity.x = 1;
        }

        int new_i = atomicAdd(n, 1);
        if (new_i < capacity){
            // printf("Particle %d spawns particle %d\n", i, new_i);
            electrons[new_i].position.y = electrons[i].position.y;
            electrons[new_i].velocity.y = electrons[i].velocity.y;
            electrons[new_i].velocity.x = -electrons[i].velocity.x;
            electrons[new_i].position.x = electrons[i].position.x + electrons[new_i].velocity.x * deltaTime;
            electrons[new_i].timestamp = t;
        }
    }
    electrons[i].position.x += electrons[i].velocity.x * deltaTime;
}

__global__ static void updateNormal(Electron* electrons, float deltaTime, int* n, int start_n, int capacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    simulate(electrons, deltaTime, n, capacity, i, 0);
}

__global__ static void updateHuge(Electron* electrons, float deltaTime, int* n, int capacity, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= *n || electrons[i].timestamp == t || electrons[i].timestamp == 0) return;

    simulate(electrons, deltaTime, n, capacity, i, t);
}

__global__ static void updateStatic(Electron* electrons, float deltaTime, int* n, int capacity, int t) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;

    for (int i = thread_id; i < *n; i += num_blocks * block_size) {
        // The thread index has passed the number of electrons. Thread returns if all electron are being handled
        if (electrons[i].timestamp == t || electrons[i].timestamp == 0) return;

        simulate(electrons, deltaTime, n, capacity, i, t);
    }
}



static void log(bool verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity){
    if (!verbose || t % 10 != 0) return;
    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(electrons_host, electrons, *n_host * sizeof(Electron), cudaMemcpyDeviceToHost);
    printf("Time %d, amount %d\n", t, *n_host);
    for(int i = 0; i < min(*n_host, capacity); i++){
        printf("%d: (%.6f, %.6f) (%.6f, %.6f)\n", i, electrons_host[i].position.x, electrons_host[i].position.y, electrons_host[i].velocity.x, electrons_host[i].velocity.y);
    }
    image(min(*n_host, capacity), electrons_host, t); // visualize a snapshot of the current positions of the particles     
    printf("\n");
}

void multiplyRun(int init_n, int capacity, int max_t, int mode, bool verbose) {
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Electron* electrons_host = (Electron *)calloc(capacity, sizeof(Electron));
    for(int i=0; i<init_n; i++) {
        electrons_host[i].position = make_float3(250, 250, 1.0);
        electrons_host[i].weight = 1.0;
        electrons_host[i].timestamp = -1;
    }

    Electron* electrons;
    cudaMalloc(&electrons, capacity * sizeof(Electron));

    cudaMemcpy(electrons, electrons_host, init_n * sizeof(Electron), cudaMemcpyHostToDevice);


    int block_size = 256;

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);

    if (verbose) printf("Time %d, amount %d\n", 0, *n_host);

    switch(mode){
        case 0: { // Normal
            printf("Multiply normal\n");
            cudaEventRecord(start);
            for (int t = 1; t < max_t; t++){
                int num_blocks = (*n_host + block_size - 1) / block_size;
                updateNormal<<<num_blocks, block_size>>>(electrons, 0.1, n, *n_host, capacity);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);

                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 1: { // Huge
            printf("Multiply huge\n");
            int num_blocks = (capacity + block_size - 1) / block_size;
            cudaEventRecord(start);
            for (int t = 1; t < max_t; t++) {
                updateHuge<<<num_blocks, block_size>>>(electrons, 0.1, n, capacity, t);
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 2: { // Static
            printf("Multiply static\n");
            int num_blocks;            
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);
            cudaEventRecord(start);
            for (int t = 1; t < max_t; t++) {
                updateStatic<<<num_blocks, block_size>>>(electrons, 0.1, n, capacity, t);
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 3: { // Dynamic
            printf("Multiply dynamic not implemented\n");
            break;
        }
        default:
            break;
    }

    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(electrons_host, electrons, *n_host * sizeof(Electron), cudaMemcpyDeviceToHost);   
    cudaEventSynchronize(stop); //skal det være her?

    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("GPU time of program: %f ms\n", runtime_ms);
}
