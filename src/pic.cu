#include <cuda_runtime.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <stdexcept>
#include "pic.h"

#define FULL_MASK 0xffffffff
#define LANE (threadIdx.x & 31)
#define WARP (threadIdx.x >> 5)

__shared__ int i_block;

__shared__ int n_block;
__shared__ int new_i_block;

__shared__ int buffer_lock;

__device__ __forceinline__ int lanemask_lt() {
    int lane = threadIdx.x & 31;
    return (1 << lane) - 1;
}

// Helper for dynamic.
__device__ void aquireLock(int* lock){
    if (LANE == 0){
        while(atomicCAS(lock, 0, 1) != 0){
            // Sleep
        }
    }
    __syncwarp();
    __threadfence_block();
}

// Helper for dynamic.
__device__ void releaseLock(int* lock){
    if (LANE == 0){
        atomicExch(lock, 0);
    }
    __threadfence_block();
}

// Helper for dynamic.
__device__ void uploadElectron(Electron* electron_dest, Electron new_electron, int test_i){
    int timestamp = new_electron.timestamp;
    new_electron.timestamp = 0;
    *electron_dest = new_electron;
    __threadfence();
    electron_dest->timestamp = timestamp;
}

// Helper for dynamic, empties buffer.
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

// The optimised scheduler, improved from Dynamic Old.
__global__ static void dynamic(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_created, int* n_done, int* i_global, float3 sim_size, CSData* d_cross_sections) {
    extern __shared__ Electron new_particles_block[];

    curandState rand_state;// = d_rand_states[thread_id];
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
        buffer_lock = 0;
    }
    __syncthreads();

    while (true){
        // If all threads in the warp have figured out that they are done, clean up and break.
        int done_mask = __ballot_sync(FULL_MASK, is_done);
        if (done_mask == FULL_MASK){
            //d_rand_states[thread_id] = rand_state;
            break;
        }

        // If WARP 0 wants to flush, do it.
        int wants_to_flush_mask = __ballot_sync(FULL_MASK, wants_to_flush);
        if (wants_to_flush_mask == FULL_MASK){
            flush_buffer(d_electrons, new_particles_block, n, capacity);
        }
        wants_to_flush = false;

        // If anyone has created a new particle, it should be added to the buffer.
        int has_new_electron_mask = __ballot_sync(FULL_MASK, has_new_electron);
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
            }
            else{  // Buffer will overflow, so flush before adding.
                if (has_new_electron && cur_n_block + rank < 32){  // Fill it completely before flushing
                    new_particles_block[cur_n_block + rank] = new_electron;
                    has_new_electron = false;
                }
                int new_i_global;
                if (LANE == 0) new_i_global = atomicAdd(n, 32);
                new_i_global = __shfl_sync(FULL_MASK, new_i_global, 0) + LANE;
                if (new_i_global < capacity){
                    uploadElectron(&d_electrons[new_i_global], new_particles_block[LANE], new_i_global);
                }
                __syncwarp();
                if (has_new_electron) {
                    new_particles_block[rank - (32 - cur_n_block)] = new_electron;
                }
                __syncwarp();
                if ((LANE == 0)) atomicExch(&n_block, new_electron_count - (32 - cur_n_block));
            }
            releaseLock(&buffer_lock);
            has_new_electron = false;
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
                if (WARP != 0 && i_local >= capacity){  // We are past the capacity and no more work can be needed. WARP 0 must stay to handle flushing.
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
                        rand_state = d_rand_states[i_local];
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
                has_new_electron = updateParticle(&electron, &new_electron, deltaTime, &rand_state, i_local, t, sim_size, d_cross_sections);
                if (has_new_electron) atomicAdd(n_created, 1);
                if (++t == over_t || electron.timestamp == DEAD){  // Particle done with this simulation
                    d_electrons[i_local] = electron;
                    d_rand_states[i_local] = rand_state;
                    t = over_t;
                    needs_new_i = true;
                    __threadfence();
                    int test = atomicAdd(n_done,1);
                }
            }
        }
    }
}

// Simulates one particle for all remaining mobility steps.
__device__ static void simulateMany(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int i, int poisson_timestep, float3 sim_size, CSData* d_cross_sections){
    Electron electron = d_electrons[i];
    curandState rand_state = d_rand_states[i];
    Electron new_electron;
    int start_t = max(1, electron.timestamp + 1);
    
    for(int t = start_t; t <= poisson_timestep; t++){
        if (updateParticle(&electron, &new_electron, deltaTime, &rand_state, i, t, sim_size, d_cross_sections)){  // True if a new particle arrived.
            if (*n < capacity){
                int new_i = atomicAdd(n, 1);
                if (new_i < capacity){
                    int timestamp = new_electron.timestamp;
                    new_electron.timestamp = 0;
                    d_electrons[new_i] = new_electron;
                    __threadfence();
                    d_electrons[new_i].timestamp = timestamp;
                }
            }
        }
        else if (electron.timestamp == DEAD){  // If particle is to be removed.
            break;
        }
    }

    d_electrons[i] = electron;
    d_rand_states[i] = rand_state;
}

// The CPU Sync scheduler as developed in the MVP (see branch final_branch).
__global__ static void cpuSync(Electron* d_electrons, double deltaTime, int* n, int start_n, int offset, int capacity, curandState* d_rand_states, int poisson_timestep, float3 sim_size, CSData* d_cross_sections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (i >= start_n) return;
    simulateMany(d_electrons, deltaTime, n, capacity, d_rand_states, i, poisson_timestep, sim_size, d_cross_sections);
}

// The naive scheduler as developed in the MVP (see branch final_branch).
__global__ static void naive(Electron* d_electrons, double deltaTime, int* n, int start_n, int capacity, curandState* d_rand_states, int t, float3 sim_size, CSData* d_cross_sections) {
    extern __shared__ Electron  new_particles_block[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) n_block = 0;

    __syncthreads();
    
    if (i >= start_n) return;

    Electron electron = d_electrons[i];
    curandState rand_state = d_rand_states[i];
    if (electron.timestamp != DEAD){
        Electron new_electron;
        if (updateParticle(&electron, &new_electron, deltaTime, &rand_state, i, t, sim_size, d_cross_sections)){
            int new_i = atomicAdd(&n_block, 1);
            new_electron.timestamp = 0;
            new_particles_block[new_i] = new_electron;
        }
        d_electrons[i] = electron;
        d_rand_states[i] = rand_state;
    }

    __syncthreads();

    if (threadIdx.x == 0){
        if (*n < capacity) new_i_block = atomicAdd(n, n_block); 
        else new_i_block = capacity;
    }

    __syncthreads();

    if (threadIdx.x >= n_block) return;
    int global_i = new_i_block + threadIdx.x;
    if (global_i >= capacity) return;
    d_electrons[global_i] = new_particles_block[threadIdx.x];
}

// The dynamic scheduler as developed in the MVP (see branch final_branch).
__global__ static void dynamicOld(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_done, int* i_global, float3 sim_size, CSData* d_cross_sections) {
    while (true) {
        __syncthreads();
        if (threadIdx.x==0) {
            i_block = atomicAdd(i_global, blockDim.x);  // Get new start index of particles for this block to simulate.
        }
        __syncthreads();

        int i = i_block + threadIdx.x;

        if (i >= capacity) break;

        while (d_electrons[i].timestamp == 0 || i >= *n) {
            // While there is no electron to work on, check if simulation is done.
            // Simulation will be done if the total amount of completely simulated particles equals the total amount of particles generated.
            int cur_n_done = atomicAdd(n_done, 0);
            __threadfence();
            int cur_n = atomicAdd(n, 0);
            if (cur_n==cur_n_done) return;
            __nanosleep(sleep_time_ns);
        }

        simulateMany(d_electrons, deltaTime, n, capacity, d_rand_states, i, poisson_timestep, sim_size, d_cross_sections);
        atomicAdd(n_done,1);  // Increment amount of fully simulated particles.
    }
}

// Removes all particles marked as dead.
// All living particles are copied to d_electrons_new.
__global__ static void remove_dead_particles(Electron* d_electrons_old, Electron* d_electrons_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool alive = (i < start_n) && (d_electrons_old[i].timestamp != DEAD);

    int alive_mask = __ballot_sync(FULL_MASK, alive);  // alive_mask shows which threads in warp have an alive particle.

    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    d_electrons_old[i].timestamp = 0;

    int i_local = 0;
    int count = __popc(alive_mask);  // Living particles in warp.
    if (count != 0){
        int leader = __ffs(alive_mask) - 1;  // First thread in warp with a living particle (though the important thing is just to agree on one thread to be leader).
        int rank = __popc(alive_mask & lanemask_lt());  // Thread's index in warp when only considering threads with living particles.

        if(LANE == leader) {
            i_local = atomicAdd(&n_block, count);  // Get warp's start index in block.
        }
        // Inform each thread in warp of their index relative to the block.
        i_local = __shfl_sync(alive_mask, i_local, leader);
        i_local += rank;
    }

    __syncthreads();
    if (threadIdx.x == 0){
        i_block = atomicAdd(n, n_block);  // Get global index for block.
    }
    __syncthreads();

    if (!alive) return;
    // Copy electron
    d_electrons_new[i_block + i_local] = d_electrons_old[i];
    d_electrons_new[i_block + i_local].timestamp = -1;
}

RunData runPIC (int init_n, int capacity, int poisson_steps, int poisson_timestep, int mode, int verbose, int block_size, int sleep_time_ns) {
    printf("PIC with\ninit n: %d\ncapacity: %d\npoisson steps: %d\npoisson_timestep: %d\nblock size: %d\nsleep time: %d\n", init_n, capacity, poisson_steps, poisson_timestep, block_size, sleep_time_ns);

    // Prepare output data
    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = poisson_steps;
    timing_data.mobility_steps = poisson_timestep;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;
    //

    double mobility_timestep = 1e-12;

    // Prepare GPU timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //

    // Read cross sections
    CSData* cross_sections = (CSData*)malloc(sizeof(CSData)*N_STEPS);
    processCSData(cross_sections, "./src/cross_section.txt");
    
    CSData* d_cross_sections;
    cudaMalloc(&d_cross_sections, N_STEPS * sizeof(CSData));
    cudaMemcpy(d_cross_sections, cross_sections, N_STEPS * sizeof(CSData), cudaMemcpyHostToDevice);
    checkCudaError("CS Alloc");
    //

    // Determine block count for some schedulers
    const int shared_mem_size_dynamic = 32 * sizeof(Electron);
    const int shared_mem_size_naive = block_size * sizeof(Electron);

    int num_blocks_pers;
    cudaDeviceGetAttribute(&num_blocks_pers, cudaDevAttrMultiProcessorCount, 0);
    int blocks_per_sm = 1;
    if (mode == 0) {
        size_t dynamics_size = shared_mem_size_dynamic + 4 * 4;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dynamic, block_size, dynamics_size);
    }
    else if (mode == 3) {
        size_t dynamics_size = 4 * 4;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, dynamicOld, block_size, dynamics_size);
    }
    printf("Multiprocessor count: %d\nBlocks per multiprocessor: %d\n", num_blocks_pers, blocks_per_sm);
    num_blocks_pers *= blocks_per_sm; 
    //

    // Prepare random numbers
    int num_blocks_rand;
    num_blocks_rand = (capacity + block_size - 1) / block_size;

    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, num_blocks_rand * block_size * sizeof(curandState));
    checkCudaError("Alloc rand");
    setup_rand<<<num_blocks_rand, block_size>>>(d_rand_states);  // This has to be done before setup_particles
    checkCudaError("Setup rand");
    //
    
    // Prepare initial particles
    Electron* h_electrons = (Electron *)calloc(capacity, sizeof(Electron));
    Electron* d_electrons;
    cudaMalloc(&d_electrons, 2 * capacity * sizeof(Electron));
    cudaMemset(d_electrons, 0, 2 * capacity * sizeof(Electron));
    checkCudaError("Electron alloc");
    setup_particles<<<(init_n + block_size - 1) / block_size, block_size>>>(&d_electrons[0], &d_rand_states[0], init_n, Sim_Size, Grid_Size);

    int* n_host = (int*)malloc(sizeof(int));
    int* n;
    cudaMalloc(&n, sizeof(int));
    *n_host = init_n;
    cudaMemcpy(n, n_host, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError("Setup particles");
    // 

    // Allocate some additional variables
    int* n_done;
    cudaMalloc(&n_done, sizeof(int));

    int* n_created;
    cudaMalloc(&n_created, sizeof(int));
    
    int* i_global;
    cudaMalloc(&i_global, sizeof(int));
    checkCudaError("Minor alloc");
    //

    // Setup grid for pic
    cudaExtent extent = make_cudaExtent(Grid_Size.x * sizeof(Cell), Grid_Size.y, Grid_Size.z);
    cudaPitchedPtr d_grid;
    cudaMalloc3D(&d_grid, extent);
    checkCudaError("Grid alloc");

    dim3 dim_block(8,8,8);
    dim3 dim_grid(Grid_Size.x/dim_block.x, Grid_Size.y/dim_block.y, Grid_Size.z/dim_block.z);
    //

    switch(mode){
        case 0:
            timing_data.function = "Dynamic";
            printf("Dynamic\n");
            break;
        case 1:
            timing_data.function = "CPU Sync";
            printf("CPU Sync\n");
            break;
        case 2:
            timing_data.function = "Naive";
            printf("Naive\n");
            break;
        case 3:
            timing_data.function = "Dynamic Old";
            printf("Dynamic Old\n");
            break;
    }

    cudaEventRecord(start);  // Start GPU timer

    int total_added = 0;
    int total_removed = 0;

    int source_index = 0;
    int destination_index = 0;

    checkCudaError("Setup");

    // Perform simulation steps.
    for (int t = 0; t < poisson_steps; t++)
    {
        // Calculate and reset variables
        int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
        source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
        destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.

        log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
        cudaMemset(n_done, 0, sizeof(int));
        cudaMemset(i_global, 0, sizeof(int));
        cudaMemcpy(n_created, n_host, sizeof(int), cudaMemcpyHostToDevice);
        //

        // Grid operations
        resetGrid<<<dim_grid, dim_block>>>(d_grid, Grid_Size);
        particlesToGrid<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
        updateGrid<<<dim_grid, dim_block>>>(d_grid, Electric_Force_Constant, Grid_Size);
        gridToParticles<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
        //

        int old_n_host = *n_host;
        // Simulate particles for this poisson step
        switch(mode){
            case 0:{  // Dynamic
                dynamic<<<num_blocks_pers, block_size, shared_mem_size_dynamic>>>(&d_electrons[source_index], mobility_timestep, n, capacity, d_rand_states, poisson_timestep, sleep_time_ns, n_created, n_done, i_global, Sim_Size, d_cross_sections);
                break;
            }
            case 1:{  // CPU Sync
                int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
                while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                    int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                    cpuSync<<<num_blocks, block_size>>>(&d_electrons[source_index], mobility_timestep, n, min(*n_host, capacity), last_n, capacity, d_rand_states, poisson_timestep, Sim_Size, d_cross_sections);
                    last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
                }
                break;
            }
            case 2:{  // Naive
                for (int mob_t = 1; mob_t <= poisson_timestep; mob_t++){
                    int num_blocks = (min(*n_host, capacity) + block_size - 1) / block_size;
                    naive<<<num_blocks, block_size, shared_mem_size_naive>>>(&d_electrons[source_index], mobility_timestep, n, min(*n_host, capacity), capacity, d_rand_states, mob_t, Sim_Size, d_cross_sections);
                    log(verbose, t, h_electrons, d_electrons, n_host, n, capacity);
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                }
                break;
            }
            case 3:{  // Dynamic Old
                dynamicOld<<<num_blocks_pers, block_size>>>(&d_electrons[source_index], mobility_timestep, n, capacity, d_rand_states, poisson_timestep, sleep_time_ns, n_done, i_global, Sim_Size, d_cross_sections);
                break;
            }
        }

        cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
        total_added += *n_host - old_n_host;
        checkCudaError("Mobility steps");

        // Check for overflow
        if (*n_host >= capacity) printf("\n\nOVERFLOW FROM ADDING PARTICLES\n\n\n");

        // Remove dead particles
        cudaMemset(n, 0, sizeof(int));
        num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
        remove_dead_particles<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], n, min(*n_host, capacity));
        old_n_host = *n_host;
        cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
        total_removed += old_n_host - *n_host;
        checkCudaError("Remove dead");
        //

        if (*n_host == 0){
            printf("Hit 0\n");
            break;
        }
    }
    log(verbose, poisson_steps, h_electrons, &d_electrons[destination_index], n_host, n, capacity);
    
    cudaEventRecord(stop);
        
    checkCudaError("After sim");

    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_electrons, &d_electrons[destination_index], min(*n_host, capacity) * sizeof(Electron), cudaMemcpyDeviceToHost);   

    cudaEventSynchronize(stop);
    float runtime_ms = 0;
    cudaEventElapsedTime(&runtime_ms, start, stop);
    printf("Final amount of particles: %d\n", min(*n_host, capacity));
    printf("Particles added: %d\n", total_added);
    printf("Particles removed: %d\n", total_removed);
    printf("GPU time of program: %f ms\n", runtime_ms);
    timing_data.time = runtime_ms;
    timing_data.final_n = min(*n_host, capacity);

    RunData run_data;
    run_data.timing_data = timing_data;
    run_data.final_n = min(*n_host, capacity);
    run_data.electrons = new Electron[min(*n_host, capacity)];
    memcpy(run_data.electrons, h_electrons, min(*n_host, capacity) * sizeof(Electron));

    free(n_host);
    free(h_electrons);
    free(cross_sections);
    cudaFree(d_electrons);
    cudaFree(d_cross_sections);
    cudaFree(n);
    cudaFree(n_created);
    cudaFree(n_done);
    cudaFree(i_global);
    cudaFree(d_rand_states);
    cudaFree(d_grid.ptr);

    return run_data;
}
