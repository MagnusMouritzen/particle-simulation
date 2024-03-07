#include <stdio.h>
#include <cuda_runtime.h>
#include "multiply_simulation.h"
#include <cooperative_groups.h>
using namespace cooperative_groups; 

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
                electrons[new_i].weight = electrons[i].weight;
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

__global__ static void updateNormalPersistentWithGlobal(Electron* electrons, float deltaTime, int* n, int start_n, int capacity, int max_t, int* wait_counter, unsigned int sleep_time_ns) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;
    int sync_agents = block_size * num_blocks;

    for(int t=1; t<=max_t; t++) {
        //if(thread_id == 0) printf("n0: %d, n1: %d, wait counter: %d, time: %d \n", n[0], n[1], *wait_counter, t);
        for (int i = thread_id; i < capacity; i += num_blocks * block_size) {
            if (thread_id >= start_n) break;
            simulate(electrons, deltaTime, n, capacity, i, t);
        }

        int dir = (t % 2) * 2 - 1;  // Alternates between -1 and 1
        int wait_target = (t % 2) * sync_agents;  // Alternates between 0 and sync_target;

        atomicAdd(&wait_counter[0], dir);
        while (atomicAdd(&wait_counter[0], 0) != wait_target) {
            __nanosleep(sleep_time_ns);
        }

        start_n = atomicAdd(n, 0);
        
        atomicAdd(&wait_counter[1], dir);
        while (atomicAdd(&wait_counter[1], 0) != wait_target){
            __nanosleep(sleep_time_ns);
        }
    }
}

__global__ static void updateNormalPersistentWithOrganisedGlobal(Electron* electrons, float deltaTime, int* n, int start_n, int capacity, int max_t, int* wait_counter, unsigned int sleep_time_ns) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;
    int sync_agents = num_blocks;

    for(int t=1; t<=max_t; t++) {
        //if(thread_id == 0) printf("n0: %d, n1: %d, wait counter: %d, time: %d \n", n[0], n[1], *wait_counter, t);
        for (int i = thread_id; i < capacity; i += num_blocks * block_size) {
            if (thread_id >= start_n) break;
            simulate(electrons, deltaTime, n, capacity, i, t);
        }

        int dir = (t % 2) * 2 - 1;  // Alternates between -1 and 1
        int wait_target = (t % 2) * sync_agents;  // Alternates between 0 and sync_target;

        
        if (threadIdx.x == 0) {
            atomicAdd(&wait_counter[0], dir);
            while (atomicAdd(&wait_counter[0], 0) != wait_target) {
                __nanosleep(sleep_time_ns);
            }
        }
        __syncthreads();

        start_n = atomicAdd(n, 0);
        
        if (threadIdx.x == 0) {
            atomicAdd(&wait_counter[1], dir);
            while (atomicAdd(&wait_counter[1], 0) != wait_target){
                __nanosleep(sleep_time_ns);
            }
        }
        __syncthreads();
    }
}

__global__ static void updateNormalPersistentWithMultiBlockSync(Electron* electrons, float deltaTime, int* n, int start_n, int capacity, int max_t) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = gridDim.x;
    int block_size = blockDim.x;
    grid_group grid = this_grid();
    
    for(int t=1; t<=max_t; t++) {
        for (int i = thread_id; i < capacity; i += num_blocks * block_size) {
            if (thread_id >= start_n) break;
            simulate(electrons, deltaTime, n, capacity, i, t);
        }
        grid.sync(); //barrier to wait for all threads in the block

        start_n = atomicAdd(n, 0);

        grid.sync();
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

__global__ static void updateDynamicThreads(Electron* electrons, float deltaTime, int* n, int capacity, int max_t, int* wait_counter, int sleep_time_ns, int* n_done, int* i_global) {

    for (int i = atomicAdd(i_global, 1); i < capacity; i = atomicAdd(i_global, 1)) {

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

__global__ static void updateDynamicBlocksAndBlockChecks(Electron* electrons, float deltaTime, int* n, int capacity, int max_t, int* wait_counter, int sleep_time_ns, int* n_done, int* i_global, int* i_blocks) {

    while (true) {
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
        __syncthreads();
        if (threadIdx.x==0) {
            atomicAdd(n_done, min(blockDim.x, capacity-i));
        }

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

TimingData multiplyRun(int init_n, int capacity, int max_t, int mode, int verbose, int block_size, int sleep_time_ns) {
    printf("Multiply with\ninit n: %d\ncapacity: %d\nmax t: %d\nblock size: %d\n", init_n, capacity, max_t, block_size);
    TimingData data;
    data.init_n = init_n;
    data.iterations = max_t;
    data.block_size = block_size;
    data.sleep_time = sleep_time_ns;

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

    int* waitCounter;
    cudaMalloc(&waitCounter, 2 * sizeof(int));
    cudaMemset(waitCounter, 0, 2 * sizeof(int));

    int* n_done;
    cudaMalloc(&n_done, sizeof(int));
    cudaMemset(n_done, 0, sizeof(int));

    int* i_global;
    cudaMalloc(&i_global, sizeof(int));
    cudaMemset(i_global, 0, sizeof(int));



    if (verbose) printf("Time %d, amount %d\n", 0, *n_host);

    switch(mode){
        case 0: { // CPU synch iterate
            printf("Multiply CPU synch iterate\n");
            data.function = "CPU synch iterate";
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
        case 1: { // Huge iterate
            printf("Multiply huge iterate\n");
            data.function = "Huge iterate";
            int num_blocks = (capacity + block_size - 1) / block_size;
            cudaEventRecord(start);
            for (int t = 1; t <= max_t; t++) {
                updateHuge<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, t);
                log(verbose, t, electrons_host, electrons, n_host, n, capacity);
            }
            cudaEventRecord(stop);
            break;
        }
        case 2: { // Static simple iterate
            printf("Multiply static simple iterate\n");
            data.function = "Static simple iterate";
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
        case 3: { // Static cooperate iterate
            printf("Multiply static cooperate iterate\n");
            data.function = "Static cooperate iterate";
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
        case 4: { // CPU Sync Full
            printf("Multiply CPU Sync Full\n");
            data.function = "CPU Sync Full";
            cudaEventRecord(start);
            int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
            while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                updateNormalFull<<<num_blocks, block_size>>>(electrons, delta_time, n, min(*n_host, capacity), last_n, capacity, max_t);
                last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
            }
            cudaEventRecord(stop);

            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);
            
            break;
        }
        case 5: { // GPU Iterate with barrier using global memory
            printf("Multiply GPU Iterate with global memory barrier \n");
            data.function = "GPU Iterate Global Memory";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            cudaEventRecord(start);

            updateNormalPersistentWithGlobal<<<num_blocks, block_size>>>(electrons, delta_time, n, init_n, capacity, max_t, waitCounter, sleep_time_ns);
            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);


            break;
        }
        case 6: { // GPU Iterate with barrier using global memory with cooperative
            printf("Multiply GPU Iterate with global memory barrier \n");
            data.function = "GPU Iterate Global Memory using cooperative";
            
            int numBlocksPerSm = 0;
            int numThreads = block_size;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, updateStatic, numThreads, 0);

            cudaEventRecord(start);
            
            dim3 dimBlock(numThreads, 1, 1);
            dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);

            void *kernelArgs[] = { &electrons, &delta_time, &n, &init_n, &capacity, &max_t, &waitCounter, &sleep_time_ns };

            cudaLaunchCooperativeKernel((void*)updateNormalPersistentWithGlobal, dimGrid, dimBlock, kernelArgs);
            
            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);


            break;
        }
        case 7: { // GPU Iterate with barrier using global memory and organised in block
            printf("Multiply GPU Iterate with global memory barrier organised\n");
            data.function = "GPU Iterate Global Memory Organised";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            cudaEventRecord(start);

            updateNormalPersistentWithOrganisedGlobal<<<num_blocks, block_size>>>(electrons, delta_time, n, init_n, capacity, max_t, waitCounter, sleep_time_ns);

            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);
            break;
        }
        case 8: { // GPU Iterate with barrier using multi block sync
            printf("Multiply GPU Iterate with multi block sync\n");
            data.function = "GPU Iterate Multi Block Sync";

            int numBlocksPerSm = 0;
            int start_n = init_n;
            // Number of threads my_kernel will be launched with
            int numThreads = block_size;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, updateStatic, numThreads, 0);

            cudaEventRecord(start);
            dim3 dimBlock(numThreads, 1, 1);
            dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);

            void *kernelArgs[] = { &electrons, &delta_time, &n, &start_n, &capacity, &max_t };

            cudaLaunchCooperativeKernel((void*)updateNormalPersistentWithMultiBlockSync, dimGrid, dimBlock, kernelArgs);
            
            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);


            break;
        }
        case 9: { // Static GPU Full
            printf("Multiply Static GPU Full\n");
            data.function = "Static GPU Full";
            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            cudaEventRecord(start);

            updateGPUIterate<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done);
            
            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);


            break;
        }
        case 10: { // Static GPU Full with cooperative
            printf("Multiply Static GPU Full\n");
            data.function = "Static GPU Full with cooperative";

            int numBlocksPerSm = 0;
            // Number of threads my_kernel will be launched with
            int numThreads = block_size;
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, updateStatic, numThreads, 0);

            cudaEventRecord(start);

            dim3 dimBlock(numThreads, 1, 1);
            dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);

            void *kernelArgs[] = {&electrons, &delta_time, &n, &capacity, &max_t, &waitCounter, &sleep_time_ns, &n_done};

            cudaLaunchCooperativeKernel((void*)updateGPUIterate, dimGrid, dimBlock, kernelArgs);

            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);


            break;
        }
        case 11: { // Dynamic with threads
            printf("Multiply dynamic with threads\n");
            data.function = "Dynamic with threads";

            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            cudaEventRecord(start);

            updateDynamicThreads<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done, i_global);
            cudaEventRecord(stop);
            
            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);

            break;
        }
        case 12: { // Dynamic with blocks
            printf("Multiply dynamic with blocks\n");
            data.function = "Dynamic with blocks";

            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            int* i_blocks;
            cudaMalloc(&i_blocks, num_blocks*sizeof(int));
            cudaMemset(i_blocks, 0, num_blocks*sizeof(int));

            cudaEventRecord(start);

            updateDynamicBlocks<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done, i_global, i_blocks);
            cudaEventRecord(stop);

            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);

            break;
        }
        case 13: { // Dynamic with blocks and block checks
            printf("Multiply dynamic with blocks and block checks\n");
            data.function = "Dynamic with blocks and block checks";

            int num_blocks;
            cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
            printf("Number of blocks: %d \n",num_blocks);

            int* i_blocks;
            cudaMalloc(&i_blocks, num_blocks*sizeof(int));
            cudaMemset(i_blocks, 0, num_blocks*sizeof(int));

            cudaEventRecord(start);

            updateDynamicBlocksAndBlockChecks<<<num_blocks, block_size>>>(electrons, delta_time, n, capacity, max_t, waitCounter, sleep_time_ns, n_done, i_global, i_blocks);
            cudaEventRecord(stop);

            log(verbose, max_t, electrons_host, electrons, n_host, n, capacity);

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
    cudaEventSynchronize(stop);
    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("GPU time of program: %f ms\n", runtime_ms);
    data.time = runtime_ms;

    free(electrons_host);
    free(n_host);
    cudaFree(electrons);
    cudaFree(n);
    cudaFree(n_done);
    cudaFree(waitCounter);

    return data;
}
