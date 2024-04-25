#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <stdexcept>

#include "pic.h"

#define FULL_MASK 0xffffffff
#define LANE (threadIdx.x & 31)
#define WARP (threadIdx.x >> 5)
#define getGridCell(x,y,z) (((Cell*)((((char*)d_grid.ptr) + z * (d_grid.pitch * grid_size.y)) + y * d_grid.pitch))[x])


__shared__ int i_block;
__shared__ int capacity;

__shared__ int n_block;
__shared__ int new_i_block;

__shared__ int buffer_lock;
__shared__ int warps_active;

__device__ __forceinline__ int lanemask_lt() {
    int lane = threadIdx.x & 31;
    return (1 << lane) - 1;
}

// These two can go to utility or something
__device__ void aquireLock(int* lock){
    if (LANE == 0){
        while(atomicCAS(lock, 0, 1) != 0){
            // Sleep
        }
    }
    __syncwarp();
}

__device__ void releaseLock(int* lock){
    if (LANE == 0){
        atomicExch(lock, 0);
    }
}

__device__ void uploadElectron(Electron* destination, Electron new_electron, int test_i){
    int timestamp = new_electron.timestamp;
    new_electron.timestamp = 0;
    *destination = new_electron;
    __threadfence();
    destination->timestamp = timestamp;
}

__device__ void flush_buffer(Electron* d_electrons, Electron* new_particles_block, int* n, int capacity){
    aquireLock(&buffer_lock);
    int cur_n_block = atomicAdd(&n_block, 0);
    if (cur_n_block != 0 && *n < capacity){  
        int new_i_global;
        if (LANE == 0) new_i_global = atomicAdd(n, cur_n_block);
        new_i_global = __shfl_sync(FULL_MASK, new_i_global, 0) + LANE;
        if (new_i_global < capacity && LANE < cur_n_block){
            uploadElectron(&d_electrons[new_i_global], new_particles_block[LANE], new_i_global);
        }
        __syncwarp();
        if ((LANE == 0)) atomicExch(&n_block, 0);
    }
    releaseLock(&buffer_lock);
}

__global__ static void poisson(Electron* d_electrons, float deltaTime, int* n, int capacity, float split_chance, float remove_chance, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_created, int* n_done, int* i_global, float3 sim_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ Electron new_particles_block[];

    curandState rand_state = d_rand_states[thread_id];
    Electron electron;
    Electron new_electron;
    bool has_new_electron = false;
    int over_t = poisson_timestep + 1;  // Past the simulation time
    int t = over_t;
    bool needs_new_i = true;
    int i_local = 0;
    bool is_done = false;
    bool wants_to_flush = false;

    if (threadIdx.x == 0) {
        n_block = 0;
        warps_active = blockDim.x / 32;
    }
    __syncthreads();

    while (true){
        // If all threads in the warp have figured out that they are done, clean up and break.
        int done_mask = __ballot_sync(FULL_MASK, is_done);
        if (done_mask == FULL_MASK){
            d_rand_states[thread_id] = rand_state;
            break;
        }

        // If WARP 0 wants to flush, do it.
        int wants_to_flush_mask = __ballot_sync(FULL_MASK, wants_to_flush);
        if (wants_to_flush_mask == FULL_MASK){
            flush_buffer(d_electrons, new_particles_block, n, capacity);
        }
        wants_to_flush = false;

        // If anyone has created a new particle, it should be added to the buffer.
        int has_new_electron_mask = __ballot_sync(~done_mask, has_new_electron);
        if (has_new_electron_mask != 0){
            int new_electron_count = __popc(has_new_electron_mask);
            int rank = __popc(has_new_electron_mask & lanemask_lt());
            aquireLock(&buffer_lock);
            int cur_n_block = atomicAdd(&n_block, 0);
            if (new_electron_count + cur_n_block <= 32){  // Buffer won't overflow, so just add
                if (has_new_electron){
                    new_particles_block[cur_n_block + rank] = new_electron;
                }
                __syncwarp();
                if (LANE == 0){
                    atomicAdd(&n_block, new_electron_count);
                }
                has_new_electron = false;
            }
            else{  // Buffer will overflow, so flush before adding.
                if (has_new_electron && cur_n_block + rank < 32){  // Fill it completely before flushing
                    new_particles_block[cur_n_block + rank] = new_electron;
                    has_new_electron = false;
                }
                if (*n < capacity){  // Hopefully this can't cause a desync
                    int new_i_global;
                    if (LANE == 0) new_i_global = atomicAdd(n, 32);
                    new_i_global = __shfl_sync(FULL_MASK, new_i_global, 0) + LANE;
                    if (new_i_global < capacity){
                        uploadElectron(&d_electrons[new_i_global], new_particles_block[LANE], new_i_global);
                    }
                }
                __syncwarp();
                if (has_new_electron) new_particles_block[rank - (32 - cur_n_block)] = new_electron;
                has_new_electron = false;
                __syncwarp();
                if ((LANE == 0)) atomicExch(&n_block, new_electron_count - (32 - cur_n_block));
            }
            releaseLock(&buffer_lock);
        }

        // Update i_local for those in need.
        int needs_new_i_mask = __ballot_sync(~done_mask, needs_new_i);
        int needs_new_i_count = __popc(needs_new_i_mask);
        if (needs_new_i){
            if (needs_new_i_count == 1){
                i_local = atomicAdd(i_global, 1);
            }
            else{
                int leader = __ffs(needs_new_i_mask) - 1;
                if (LANE == leader){
                    i_local = atomicAdd(i_global, needs_new_i_count);
                }
                int rank = __popc(needs_new_i_mask & lanemask_lt());
                i_local = __shfl_sync(needs_new_i_mask, i_local, leader) + rank;
            }
            needs_new_i = false;
        }

        // Update particle, look for new work, or check for done.
        if (!is_done){
            if (t == over_t){  // We don't have a particle and must load a new one
                if (WARP != 0 && i_local >= capacity){
                    is_done = true;
                }
                else if (i_local < min(*n, capacity) && d_electrons[i_local].timestamp != 0){  // Check if the particle we are looking at is ready yet
                    if (d_electrons[i_local].timestamp == over_t-1){  // This new electron was spawned at the very end and needs no more.
                        needs_new_i = true;
                        __threadfence();
                        atomicAdd(n_done,1);
                    }
                    else{
                        electron = d_electrons[i_local];
                        t =  max(1, electron.timestamp + 1);
                    }
                }
                else{  // Check if we are done with the simulation
                    int cur_n_done = *n_done;
                    __threadfence();
                    int cur_n_created = min(capacity, *n_created);
                    if (cur_n_created == cur_n_done) {
                        is_done = true;
                    }
                    else if (WARP == 0){
                        // There might be some particles left in a buffer. 
                        // It might as well just be one warp to deal with it.
                        if (*n== cur_n_done && n_block != 0){
                            wants_to_flush = true;
                        }
                    }
                }
            }

            if (t != over_t)  // Check needed again because we might not have loaded a new particle
            {
                has_new_electron = updateParticle(&electron, &new_electron, deltaTime, split_chance, remove_chance, &rand_state, i_local, t, sim_size);
                if (has_new_electron) atomicAdd(n_created, 1);
                if (++t == over_t || electron.timestamp == DEAD){  // Particle done with this simulation
                    d_electrons[i_local] = electron;
                    t = over_t;
                    needs_new_i = true;
                    __threadfence();
                    int test = atomicAdd(n_done,1);
                }
            }
        }
    }

    // If we are the last warp to quite, flush buffer to ensure nothing is forgotten.
    int leaving_index;
    if (LANE == 0) leaving_index = atomicAdd(&warps_active, -1);
    leaving_index = __shfl_sync(FULL_MASK, leaving_index, 0);

    if (leaving_index == 1){
        flush_buffer(d_electrons, new_particles_block, n, capacity);
    }
}

__global__ static void remove_dead_particles(Electron* d_electrons_old, Electron* d_electrons_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // int mask = __ballot_sync(FULL_MASK, i >= start_n);
    bool alive = (i < start_n) && (d_electrons_old[i].timestamp != DEAD);
    int alive_mask = __ballot_sync(FULL_MASK, alive);

    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    int count = __popc(alive_mask);
    int leader = __ffs(alive_mask) - 1;
    int rank = __popc(alive_mask & lanemask_lt());

    int i_local = 0;
    if(LANE == leader) {
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
    d_electrons_new[i_block + i_local].timestamp = -1;
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

    int num_blocks_pers;
    cudaDeviceGetAttribute(&num_blocks_pers, cudaDevAttrMultiProcessorCount, 0);

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

    int* n_created;
    cudaMalloc(&n_created, sizeof(int));
    cudaMemcpy(n_created, n_host, sizeof(int), cudaMemcpyHostToDevice);
    
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
            const int sharedMemSize = 32 * sizeof(Electron);
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

                poisson<<<num_blocks_pers, block_size, sharedMemSize>>>(&d_electrons[source_index], 0.0001, n, capacity, split_chance, remove_chance, d_rand_states, poisson_timestep, sleep_time_ns, n_created, n_done, i_global, Sim_Size);
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                checkCudaError("Poisson");
                cudaMemset(n, 0, sizeof(int));
                num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
                remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
                cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(n_created, n_host, sizeof(int), cudaMemcpyHostToDevice);
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