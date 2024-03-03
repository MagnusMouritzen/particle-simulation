#include <cuda_runtime.h>
#include <stdio.h>
#include "multiply_simulation.h"

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

__global__ static void updateNormal(Electron* electrons, float deltaTime, int* n, int start_n, int capacity, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    simulate(electrons, deltaTime, n, capacity, i, t);
}

__global__ static void updateHuge(Electron* electrons, float deltaTime, int* n, int capacity, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= min(*n, capacity) || electrons[i].timestamp == t || electrons[i].timestamp == 0) return;

    simulate(electrons, deltaTime, n, capacity, i, t);
}

__global__ static void updateStatic(Electron* electrons, float deltaTime, int* n, int capacity, int t) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;

    for (int i = thread_id; i < min(*n, capacity); i += num_blocks * block_size) {
        // The thread index has passed the number of electrons. Thread returns if all electron are being handled
        if (electrons[i].timestamp == t || electrons[i].timestamp == 0) return;

        simulate(electrons, deltaTime, n, capacity, i, t);
    }
}

__global__ static void updateNormalFull(Electron* electrons, float deltaTime, int* n, int start_n, int offset, int capacity, int max_t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    
    // The thread index has passed the number of electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    for(int t = max(1, electrons[i].timestamp + 1); t <= max_t; t++){
        simulate(electrons, deltaTime, n, capacity, i, t);
    }
}


static void log(int verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity){
    if (verbose == 0 || t % verbose != 0) return;
    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    int true_n = min(*n_host, capacity);
    cudaMemcpy(electrons_host, electrons, true_n * sizeof(Electron), cudaMemcpyDeviceToHost);
    printf("Time %d, amount %d\n", t, *n_host);
    for(int i = 0; i < true_n; i++){
        printf("%d: (%.6f, %.6f) (%.6f, %.6f) [%d]\n", i, electrons_host[i].position.x, electrons_host[i].position.y, electrons_host[i].velocity.x, electrons_host[i].velocity.y, electrons_host[i].timestamp);
    }
    image(true_n, electrons_host, t); // visualize a snapshot of the current positions of the particles     
    printf("\n");
}

TimingData multiplyRun(int init_n, int capacity, int max_t, int mode, int verbose, int block_size) {
    printf("Multiply with\ninit n: %d\ncapacity: %d\nmax t: %d\nblock size: %d\n", init_n, capacity, max_t, block_size);
    TimingData data;
    data.init_n = init_n;
    data.iterations = max_t;
    data.block_size = block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    Electron* electrons_host = (Electron *)calloc(capacity, sizeof(Electron));
    for(int i=0; i<init_n; i++) {
        electrons_host[i].position = make_float3(250, 250, 1.0);
        electrons_host[i].weight = 1.0;
        electrons_host[i].timestamp = -1;
    }

    float delta_time = 0.1;

    Electron* electrons;
    cudaMalloc(&electrons, capacity * sizeof(Electron));

    cudaMemcpy(electrons, electrons_host, init_n * sizeof(Electron), cudaMemcpyHostToDevice);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);

    if (verbose) printf("Time %d, amount %d\n", 0, *n_host);

    switch(mode){
        case 0: { // Normal
            printf("Multiply normal\n");
            data.function = "Normal";
            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++){
                int num_blocks = (min(*n_host, capacity) + block_size - 1) / block_size;
                updateNormal<<<num_blocks, block_size>>>(electrons, delta_time, n, min(*n_host, capacity), capacity, t);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);

                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 1: { // Huge
            printf("Multiply huge\n");
            data.function = "Huge";
            int num_blocks = (capacity + block_size - 1) / block_size;
            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++) {
                updateHuge<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, t);
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 2: { // Static simple
            printf("Multiply static simple\n");
            data.function = "Static simple";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);
            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++) {
                updateStatic<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, t);
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 3: { // Static advanced
            printf("Multiply static advanced\n");
            data.function = "Static advanced";
            int numBlocksPerSm = 0;
            // Number of threads my_kernel will be launched with
            int numThreads = block_size;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);  // What number should this actually be?
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, updateStatic, numThreads, 0);
            // launch
            dim3 dimBlock(numThreads, 1, 1);
            dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
            printf("numBlocksPerSm: %d \n",numBlocksPerSm);
            printf("multiProcessorCount: %d \n",deviceProp.multiProcessorCount);

            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++) {
                void *kernelArgs[] = { &electrons, &delta_time, &n, &capacity, &t };
                cudaLaunchCooperativeKernel((void*)updateStatic, dimGrid, dimBlock, kernelArgs);
                cudaDeviceSynchronize();
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 4: { // Normal full
            printf("Multiply normal full\n");
            data.function = "Normal full";
            cudaEventRecord(start);
            int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
            while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                updateNormalFull<<<num_blocks, block_size>>>(electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, max_t);
                last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
            }

            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);
            
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

    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(electrons_host, electrons, min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);   
    cudaEventSynchronize(stop); //skal det være her?

    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("GPU time of program: %f ms\n", runtime_ms);
    data.time = runtime_ms;
    return data;
}
