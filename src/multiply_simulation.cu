#include <cuda_runtime.h>
#include <stdio.h>
#include "utility.h"
#include "multiply_simulation.h"



__global__ static void updateNormal(Electron* electrons, float deltaTime, int* n, int start_n, int capacity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start_n) return;
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
            printf("Particle %d spawns particle %d\n", i, new_i);
            electrons[new_i].position.y = electrons[i].position.y;
            electrons[new_i].velocity.y = electrons[i].velocity.y;
            electrons[new_i].velocity.x = -electrons[i].velocity.x;
            electrons[new_i].position.x = electrons[i].position.x + electrons[new_i].velocity.x * deltaTime;
        }
    }
    electrons[i].position.x += electrons[i].velocity.x * deltaTime;
}

__global__ static void updateHuge(Electron* electrons, float deltaTime) {
    
}

__global__ static void updateStatic(Electron* electrons, float deltaTime) {
}

__global__ static void updateDynamic(Electron* electrons, float deltaTime) {
}


void multiplyRun(int init_n, int capacity, int max_t, int mode, bool verbose) {
    
    Electron* electrons_host = (Electron *)malloc(capacity * sizeof(Electron));
    for(int i=0; i<init_n; i++) {
        electrons_host[i].position = make_float3(25, 5, 1.0);
        electrons_host[i].weight = 1.0;
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
        case 0:
            // Normal
            for (int t = 1; t < max_t; t++){
                int num_blocks = (*n_host + block_size - 1) / block_size;
                updateNormal<<<num_blocks, block_size>>>(electrons, 0.1, n, *n_host, capacity);
                
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);

                if (verbose && t % 1 == 0){
                    cudaMemcpy(electrons_host, electrons, *n_host * sizeof(Electron), cudaMemcpyDeviceToHost);
                    printf("Time %d, amount %d\n", t, *n_host);
                    for(int i = 0; i < min(*n_host, capacity); i++){
                        if (i >= capacity) break;
                        printf("%d: (%.6f, %.6f) (%.6f, %.6f)\n", i, electrons_host[i].position.x, electrons_host[i].position.y, electrons_host[i].velocity.x, electrons_host[i].velocity.y);
                    }
                    image(min(*n_host, capacity), electrons_host, t); // visualize a snapshot of the current positions of the particles     
                    printf("\n");
                }
                if (*n_host >= capacity) break;
            }
            break;
        case 1:
            // Huge
            break;
        case 2:
            // Static
            break;
        case 3:
            // Dynamic
            break;
        default:
            break;
    }

    cudaMemcpy(electrons_host, electrons, *n_host * sizeof(Electron), cudaMemcpyDeviceToHost);   
}
