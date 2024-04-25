#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include "pic.h"

#define FULL_MASK 0xffffffff


__shared__ int i_block;
__shared__ int capacity;

__shared__ int n_block;
__shared__ int new_i_block;


#define getGridCell(x,y,z) (((Cell*)((((char*)d_grid.ptr) + z * (d_grid.pitch * grid_size.y)) + y * d_grid.pitch))[x])
__device__ __forceinline__ int lanemask_lt() {
    int lane = threadIdx.x & 31;
    return (1 << lane) - 1;
}

__device__ static void simulateMany(Electron* d_electrons, float deltaTime, int* n, int capacity, curandState* rand_state, int i, int start_t, int poisson_timestep, float3 sim_size, CSData* d_cross_sections){
    Electron electron = d_electrons[i];

    for(int t = start_t; t <= poisson_timestep; t++){
        int new_i = updateParticle(&electron, d_electrons, deltaTime, n, capacity, rand_state, i, t, sim_size, d_cross_sections);
        if(new_i != -1 && new_i < capacity) {  // If a new particle was spawned and there is room for it.
            __threadfence();
            d_electrons[new_i].timestamp = t;
            // printf("%d: (%d) NEW %d {%f}\n", i, t, new_i, d_electrons[new_i].position.x);
        }
        else if (electron.timestamp == DEAD){  // If particle is to be removed.
            // printf("%d: (%d) DEAD\n", i, t);
            break;
        }
    }
    if (electron.timestamp != DEAD) electron.timestamp = -1;

    d_electrons[i] = electron;
}

__global__ static void poisson(Electron* d_electrons, float deltaTime, int* n, int capacity, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_done, int* i_global, float3 sim_size, CSData* d_cross_sections) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {
        __syncthreads(); //sync threads seems to be able to handle threads being terminated
        if (threadIdx.x==0) {
            i_block = atomicAdd(i_global, blockDim.x);
        }
        __syncthreads();

        int i = i_block + threadIdx.x;

        if (i >= capacity) break;

        while (d_electrons[i].timestamp == 0 || i >= *n) {
            int cur_n_done = *n_done;
            __threadfence();
            int cur_n = *n;
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        simulateMany(d_electrons, deltaTime, n, capacity, &d_rand_states[thread_id], i, max(1, d_electrons[i].timestamp + 1), poisson_timestep, sim_size, d_cross_sections);
        atomicAdd(n_done,1);

    }
}

__global__ static void remove_dead_particles(Electron* d_electrons_old, Electron* d_electrons_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool alive = (i < start_n) && (d_electrons_old[i].timestamp != DEAD);
    int alive_mask = __ballot_sync(FULL_MASK, alive);

    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    int count = __popc(alive_mask);
    int leader = __ffs(alive_mask) - 1;
    int rank = __popc(alive_mask & lanemask_lt());

    int i_local = 0;
    if((threadIdx.x & 31) == leader) {
        i_local = atomicAdd(&n_block, count);
    }
    i_local = __shfl_sync(alive_mask, i_local, leader);
    i_local += rank;

    __syncthreads();
    if (threadIdx.x == 0){
        i_block = atomicAdd(n, n_block);
    }
    __syncthreads();

    if (!alive) return;
    d_electrons_new[i_block + i_local] = d_electrons_old[i];
}

__global__ static void particlesToGrid(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;


    Electron electron = d_electrons[i];

    int x = electron.position.x/cell_size;
    int y = electron.position.y/cell_size;
    int z = electron.position.z/cell_size;

    atomicAdd(&getGridCell(x,y,z).charge, 1);

}

__global__ static void gridToParticles(cudaPitchedPtr d_grid, Electron* d_electrons, int* n, int3 grid_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *n) return;

    Electron electron = d_electrons[i];

    int x = electron.position.x/cell_size;
    int y = electron.position.y/cell_size;
    int z = electron.position.z/cell_size;


    electron.acceleration =  getGridCell(x,y,z).acceleration;


    d_electrons[i] = electron;

}

__global__ void resetGrid(cudaPitchedPtr d_grid, int3 grid_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    getGridCell(x,y,z).charge = 0;
}

__global__ void updateGrid(cudaPitchedPtr d_grid, double electric_force_constant, int3 grid_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // INSERT ELEMENT TO 3D ARRAY
    char* gridPtr = (char*)d_grid.ptr;
    size_t pitch = d_grid.pitch; // the number of bytes in a row of the array
    size_t slicePitch = pitch * grid_size.y; // the number of bytes pr slice

    char* slice = gridPtr + z * slicePitch; // get slice 

    char* row = (slice + y * pitch); // get row in slice


    double xAcc = 0;
    if (x != 0) xAcc -= ((Cell*)row)[x-1].charge;
    if (x != grid_size.x-1) xAcc += ((Cell*)row)[x+1].charge;
    xAcc *= electric_force_constant;

    double yAcc = 0;
    if (y != 0) yAcc -= ((Cell*)(row - pitch))[x].charge;
    if (y != grid_size.y-1) yAcc += ((Cell*)(row + pitch))[x].charge;
    yAcc *= electric_force_constant;

    double zAcc = 0;
    if (z != 0) zAcc -= ((Cell*)(row-slicePitch))[x].charge;
    if (z != grid_size.z-1) zAcc += ((Cell*)(row+slicePitch))[x].charge;
    zAcc *= electric_force_constant;

    ((Cell*)row)[x].acceleration = make_float3((float)xAcc, (float)yAcc, (float)zAcc);

}

__global__ static void naive(Electron* d_electrons, float deltaTime, int* n, int start_n, int capacity, curandState* d_rand_states, int t, float3 sim_size, CSData* d_cross_sections) {

    extern __shared__ Electron  new_particles_block[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) n_block = 0;

    __syncthreads(); // Ensure construction is finished
    
    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    updateParticle(&d_electrons[i], new_particles_block, deltaTime, &n_block, capacity, &d_rand_states[i], i, t, sim_size, d_cross_sections);

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
    d_electrons[global_i].timestamp = t;
}

__global__ static void cpuSynch(Electron* d_electrons, float deltaTime, int* n, int start_n, int offset, int capacity, curandState* d_rand_states, int poisson_timestep, float3 sim_size, CSData* d_cross_sections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    simulateMany(d_electrons, deltaTime, n, capacity, &d_rand_states[i], i, max(1, d_electrons[i].timestamp + 1), poisson_timestep, sim_size, d_cross_sections);
}

RunData runPIC (int init_n, int capacity, int poisson_steps, int poisson_timestep, int mode, int verbose, int block_size, int sleep_time_ns) {
    printf("MVP with\ninit n: %d\ncapacity: %d\npoisson steps: %d\npoisson_timestep: %d\nblock size: %d\nsleep time: %d\n", init_n, capacity, poisson_steps, poisson_timestep, block_size, sleep_time_ns);

    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = poisson_steps;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CSData* cross_sections = (CSData*)malloc(sizeof(CSData)*11);
    ProcessCSData(cross_sections, 11, "/zhome/b5/3/156408/Desktop/particle-simulation/src/cross_section.txt");

    
    CSData* d_cross_sections;
    cudaMalloc(&d_cross_sections, 11 * sizeof(CSData));
    cudaMemcpy(d_cross_sections, cross_sections, 11 * sizeof(CSData), cudaMemcpyHostToDevice);

    int num_blocks;
    int num_blocks_pers;
    size_t dynamics_size = 16;
    cudaDeviceGetAttribute(&num_blocks, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_pers, poisson, block_size, dynamics_size);
    num_blocks_pers *= num_blocks; 

    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, num_blocks_pers * block_size * sizeof(curandState));
    setup_rand<<<num_blocks_pers, block_size>>>(d_rand_states);  // This has to be done before setup_particles
    
    Electron* h_electrons = (Electron *)calloc(capacity, sizeof(Electron));
    Electron* d_electrons;
    cudaMalloc(&d_electrons, 2 * capacity * sizeof(Electron));
    cudaMemset(d_electrons, 0, 2 * capacity * sizeof(Electron));
    setup_particles<<<(init_n + block_size - 1) / block_size, block_size>>>(d_electrons, d_rand_states, init_n, Sim_Size, Grid_Size);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);

    int* n_done;
    cudaMalloc(&n_done, sizeof(int));
    
    int* i_global;
    cudaMalloc(&i_global, sizeof(int));


    cudaExtent extent = make_cudaExtent(Grid_Size.x * sizeof(Cell), Grid_Size.y, Grid_Size.z);
    cudaPitchedPtr d_grid;
    cudaMalloc3D(&d_grid, extent);

    dim3 dim_block(8,8,8);
    dim3 dim_grid(Grid_Size.x/dim_block.x, Grid_Size.y/dim_block.y, Grid_Size.z/dim_block.z);

    switch(mode){
        case 0: { // GOOD
            timing_data.function = "GOOD";
            cudaEventRecord(start);

            int source_index = 0;
            int destination_index = 0;
            for (int t = 0; t < poisson_steps; t++)
            {
                int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
                destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.

                log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
                cudaMemset(n_done, 0, sizeof(int));
                cudaMemset(i_global, 0, sizeof(int));

                resetGrid<<<dim_grid, dim_block>>>(d_grid, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Reset grid");
                particlesToGrid<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Particles to grid");
                updateGrid<<<dim_grid, dim_block>>>(d_grid, Electric_Force_Constant, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Update grid");
                gridToParticles<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Grid to particles");

                poisson<<<num_blocks_pers, block_size>>>(&d_electrons[source_index], 0.0001, n, capacity, d_rand_states, poisson_timestep, sleep_time_ns, n_done, i_global, Sim_Size, d_cross_sections);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError("Poisson");
                cudaMemset(n, 0, sizeof(int));
                num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError("Remove dead");
                if (*n_host == 0){
                    printf("Hit 0\n");
                    break;
                }
            }
            log(verbose, poisson_steps, h_electrons, &d_electrons[destination_index], n_host, n, capacity);
            
            
            cudaEventRecord(stop);
            break;
        }
        case 1: {
            timing_data.function = "CPU Sync";
            cudaEventRecord(start);

            int source_index = 0;
            int destination_index = 0;
            for (int t = 0; t < poisson_steps; t++) {

                int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
                destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.

                log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
                cudaMemset(n_done, 0, sizeof(int));
                cudaMemset(i_global, 0, sizeof(int));

                resetGrid<<<dim_grid, dim_block>>>(d_grid, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Reset grid");
                particlesToGrid<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Particles to grid");
                updateGrid<<<dim_grid, dim_block>>>(d_grid, Electric_Force_Constant, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Update grid");
                gridToParticles<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Grid to particles");


                int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
                while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                    int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                    cpuSynch<<<num_blocks, block_size>>>(&d_electrons[source_index], 0.0001, n, min(*n_host, capacity), last_n, capacity, d_rand_states, poisson_timestep, Sim_Size, d_cross_sections);
                    last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
                }

                checkCudaError("Poisson");
                cudaMemset(n, 0, sizeof(int));
                num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError("Remove dead");
                if (*n_host == 0) {
                    printf("Hit 0\n");
                    break;
                }
            }
            cudaEventRecord(stop);
            break;
        }
        case 2: {
            timing_data.function = "Naive";
            const int sharedMemSize = block_size * sizeof(Electron);
            cudaEventRecord(start);

            int source_index = 0;
            int destination_index = 0;
            for (int t = 0; t < poisson_steps; t++) {
                int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
                destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.

                log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
                cudaMemset(n_done, 0, sizeof(int));
                cudaMemset(i_global, 0, sizeof(int));

                resetGrid<<<dim_grid, dim_block>>>(d_grid, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Reset grid");
                particlesToGrid<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Particles to grid");
                updateGrid<<<dim_grid, dim_block>>>(d_grid, Electric_Force_Constant, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Update grid");
                gridToParticles<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
                checkCudaError("Grid to particles");


                for (int t = 1; t <= poisson_timestep; t++){
                    int num_blocks = (min(*n_host, capacity) + block_size - 1) / block_size;
                    naive<<<num_blocks, block_size, sharedMemSize>>>(&d_electrons[source_index], 0.0001, n, min(*n_host, capacity), capacity, d_rand_states, t, Sim_Size, d_cross_sections);
                    log(verbose, t, h_electrons, d_electrons, n_host, n, capacity);
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                }

                checkCudaError("Poisson");
                cudaMemset(n, 0, sizeof(int));
                num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError("Remove dead");
                if (*n_host == 0) {
                    printf("Hit 0\n");
                    break;
                }
            }

            cudaEventRecord(stop);
            break;
        }
        default:
            break;
    }
    checkCudaError("After sim");

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


// LOOP OVER 3D ARRAY
// char* gridPtr = d_grid.ptr;
// size_t pitch = d_grid.pitch; // the number of bytes in a row of the array
// size_t slicePitch = pitch * grid_size.y; // the number of bytes pr slice
// for (int z = 0; z < grid_size.z; ++z) {
//     char* slice = gridPtr + z * slicePitch;
//     for (int y = 0; y < grid_size.y; ++y) {
//         Cell* row = (Cell*)(slice + y * pitch);
//         for (int x = 0; x < grid_size.x; ++x) {
//             Cell element = row[x];
//         }
//     }
// }


// INSERT ELEMENT TO 3D ARRAY
// char* gridPtr = d_grid.ptr;
// size_t pitch = d_grid.pitch; // the number of bytes in a row of the array
// size_t slicePitch = pitch * grid_size.y; // the number of bytes pr slice

// char* slice = gridPtr + z * slicePitch; // get slice 

// Cell* row = (Cell*)(slice + y * pitch); // get row in slice

// row[x].charge = 0;