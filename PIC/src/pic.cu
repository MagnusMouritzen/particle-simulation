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

// These two can go to utility or something
__device__ void aquireLock(int* lock){
    if (LANE == 0){
        while(atomicCAS(lock, 0, 1) != 0){
            // Sleep
        }
    }
    __syncwarp();
    __threadfence_block();
}

__device__ void releaseLock(int* lock){
    if (LANE == 0){
        atomicExch(lock, 0);
    }
    __threadfence_block();
}

__device__ void uploadElectron(Electron* electron_dest, curandState* rand_state_dest, Electron new_electron, curandState new_rand_state, int test_i){
    int timestamp = new_electron.timestamp;
    new_electron.timestamp = 0;
    *electron_dest = new_electron;
    *rand_state_dest = new_rand_state;
    __threadfence();
    electron_dest->timestamp = timestamp;
}

__device__ void flush_buffer(Electron* d_electrons, curandState* d_rand_states, Electron* new_particles_block, curandState* new_rand_states_block, int* n, int capacity){
    aquireLock(&buffer_lock);
    int cur_n_block = atomicAdd(&n_block, 0);
    if (cur_n_block != 0 && *n < capacity){  
        int new_i_global;
        if (LANE == 0) new_i_global = atomicAdd(n, cur_n_block);
        new_i_global = __shfl_sync(FULL_MASK, new_i_global, 0) + LANE;
        if (new_i_global < capacity && LANE < cur_n_block){
            uploadElectron(&d_electrons[new_i_global], &d_rand_states[new_i_global], new_particles_block[LANE], new_rand_states_block[LANE], new_i_global);
        }
        __syncwarp();
        if ((LANE == 0)) atomicExch(&n_block, 0);
    }
    releaseLock(&buffer_lock);
}

__global__ static void dynamic(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_created, int* n_done, int* i_global, float3 sim_size, CSData* d_cross_sections) {
    //int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ char shared_memory[];
    Electron* new_particles_block = reinterpret_cast<Electron*>(shared_memory);
    curandState* new_rand_states_block = reinterpret_cast<curandState*>(shared_memory + 32 * sizeof(Electron));

    curandState rand_state;// = d_rand_states[thread_id];
    Electron electron;
    Electron new_electron;
    curandState new_rand_state;
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
            flush_buffer(d_electrons, d_rand_states, new_particles_block, new_rand_states_block, n, capacity);
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
                    new_rand_states_block[cur_n_block + rank] = new_rand_state;
                }
                __syncwarp();
                if (LANE == 0){
                    atomicAdd(&n_block, new_electron_count);
                }
            }
            else{  // Buffer will overflow, so flush before adding.
                if (has_new_electron && cur_n_block + rank < 32){  // Fill it completely before flushing
                    new_particles_block[cur_n_block + rank] = new_electron;
                    new_rand_states_block[cur_n_block + rank] = new_rand_state;
                    has_new_electron = false;
                }
                int new_i_global;
                if (LANE == 0) new_i_global = atomicAdd(n, 32);
                new_i_global = __shfl_sync(FULL_MASK, new_i_global, 0) + LANE;
                if (new_i_global < capacity){
                    uploadElectron(&d_electrons[new_i_global], &d_rand_states[new_i_global], new_particles_block[LANE], new_rand_states_block[LANE], new_i_global);
                }
                __syncwarp();
                if (has_new_electron) {
                    new_particles_block[rank - (32 - cur_n_block)] = new_electron;
                    new_rand_states_block[rank - (32 - cur_n_block)] = new_rand_state;
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
                has_new_electron = updateParticle(&electron, &new_electron, &new_rand_state, deltaTime, &rand_state, i_local, t, sim_size, d_cross_sections);
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

__device__ static void simulateMany(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int i, int poisson_timestep, float3 sim_size, CSData* d_cross_sections){
    Electron electron = d_electrons[i];
    curandState rand_state = d_rand_states[i];
    Electron new_electron;
    curandState new_rand_state;
    int start_t = max(1, electron.timestamp + 1);
    
    for(int t = start_t; t <= poisson_timestep; t++){
        if (updateParticle(&electron, &new_electron, &new_rand_state, deltaTime, &rand_state, i, t, sim_size, d_cross_sections)){
            if (*n < capacity){
                int new_i = atomicAdd(n, 1);
                if (new_i < capacity){
                    int timestamp = new_electron.timestamp;
                    new_electron.timestamp = 0;
                    d_electrons[new_i] = new_electron;
                    d_rand_states[new_i] = new_rand_state;
                    __threadfence();
                    d_electrons[new_i].timestamp = timestamp;
                }
            }
        }
        else if (electron.timestamp == DEAD){  // If particle is to be removed.
            // printf("%d: (%d) DEAD\n", i, t);
            break;
        }
    }

    d_electrons[i] = electron;
    d_rand_states[i] = rand_state;
}

__global__ static void cpuSync(Electron* d_electrons, double deltaTime, int* n, int start_n, int offset, int capacity, curandState* d_rand_states, int poisson_timestep, float3 sim_size, CSData* d_cross_sections) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;

    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;
    simulateMany(d_electrons, deltaTime, n, capacity, d_rand_states, i, poisson_timestep, sim_size, d_cross_sections);
}

__global__ static void naive(Electron* d_electrons, double deltaTime, int* n, int start_n, int capacity, curandState* d_rand_states, int t, float3 sim_size, CSData* d_cross_sections) {
    extern __shared__ char shared_memory[];
    Electron* new_particles_block = reinterpret_cast<Electron*>(shared_memory);
    curandState* new_rand_states_block = reinterpret_cast<curandState*>(shared_memory + blockDim.x * sizeof(Electron));

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) n_block = 0;

    __syncthreads(); // Ensure construction is finished
    
    // The thread index has passed the number of d_electrons. Thread returns if all electron are being handled
    if (i >= start_n) return;

    Electron electron = d_electrons[i];
    curandState rand_state = d_rand_states[i];
    if (electron.timestamp != DEAD){
        Electron new_electron;
        curandState new_rand_state;
        if (updateParticle(&electron, &new_electron, &new_rand_state, deltaTime, &rand_state, i, t, sim_size, d_cross_sections)){
            int new_i = atomicAdd(&n_block, 1);
            new_electron.timestamp = 0;
            new_particles_block[new_i] = new_electron;
            new_rand_states_block[new_i] = new_rand_state;
        }
        d_electrons[i] = electron;
        d_rand_states[i] = rand_state;
    }

    __syncthreads();

    if (threadIdx.x == 0){
        if (*n < capacity) new_i_block = atomicAdd(n, n_block);  // Avoid risk of n overflowing int max value
        else new_i_block = capacity;
    }

    __syncthreads();

    if (threadIdx.x >= n_block) return;
    int global_i = new_i_block + threadIdx.x;
    if (global_i >= capacity) return;
    d_electrons[global_i] = new_particles_block[threadIdx.x];
    d_rand_states[global_i] = new_rand_states_block[threadIdx.x];
}

__global__ static void dynamicOld(Electron* d_electrons, double deltaTime, int* n, int capacity, curandState* d_rand_states, int poisson_timestep, int sleep_time_ns, int* n_done, int* i_global, float3 sim_size, CSData* d_cross_sections) {
    //int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
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

        simulateMany(d_electrons, deltaTime, n, capacity, d_rand_states, i, poisson_timestep, sim_size, d_cross_sections);
        atomicAdd(n_done,1);
    }
}

__global__ static void remove_dead_particles(Electron* d_electrons_old, Electron* d_electrons_new, curandState* d_rand_states_old, curandState* d_rand_states_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool alive = (i < start_n) && (d_electrons_old[i].timestamp != DEAD);
    int alive_mask = __ballot_sync(FULL_MASK, alive);

    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    d_electrons_old[i].timestamp = 0;

    int i_local = 0;
    int count = __popc(alive_mask);
    if (count != 0){
        int leader = __ffs(alive_mask) - 1;
        int rank = __popc(alive_mask & lanemask_lt());

        if(LANE == leader) {
            i_local = atomicAdd(&n_block, count);
        }
        i_local = __shfl_sync(alive_mask, i_local, leader);
        i_local += rank;
    }

    __syncthreads();
    if (threadIdx.x == 0){
        i_block = atomicAdd(n, n_block);
    }
    __syncthreads();

    if (!alive) return;
    d_rand_states_new[i_block + i_local] = d_rand_states_old[i];
    d_electrons_new[i_block + i_local] = d_electrons_old[i];
    d_electrons_new[i_block + i_local].timestamp = -1;
}

__global__ static void remove_dead_particles_simp(Electron* d_electrons_old, Electron* d_electrons_new, curandState* d_rand_states_old, curandState* d_rand_states_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= start_n) return;
    bool alive = d_electrons_old[i].timestamp != DEAD;

    d_electrons_old[i].timestamp = 0;

    if (!alive) return;

    int i_new = atomicAdd(n, 1);
    d_rand_states_new[i_new] = d_rand_states_old[i];
    d_electrons_new[i_new] = d_electrons_old[i];
    d_electrons_new[i_new].timestamp = -1;
}

__global__ static void remove_dead_particles_block(Electron* d_electrons_old, Electron* d_electrons_new, curandState* d_rand_states_old, curandState* d_rand_states_new, int* n, int start_n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool alive = (i < start_n) && (d_electrons_old[i].timestamp != DEAD);

    if (i >= start_n) return;

    if (threadIdx.x == 0) n_block = 0;
    __syncthreads();

    d_electrons_old[i].timestamp = 0;

    int i_local;
    if (alive) i_local = atomicAdd(&n_block, 1);

    __syncthreads();
    if (threadIdx.x == 0){
        i_block = atomicAdd(n, n_block);
    }
    __syncthreads();

    if (!alive) return;
    d_rand_states_new[i_block + i_local] = d_rand_states_old[i];
    d_electrons_new[i_block + i_local] = d_electrons_old[i];
    d_electrons_new[i_block + i_local].timestamp = -1;
}

RunData runPIC (int init_n, int capacity, int poisson_steps, int poisson_timestep, int mode, int verbose, int block_size, int sleep_time_ns) {
    printf("PIC with\ninit n: %d\ncapacity: %d\npoisson steps: %d\npoisson_timestep: %d\nblock size: %d\nsleep time: %d\n", init_n, capacity, poisson_steps, poisson_timestep, block_size, sleep_time_ns);

    TimingData timing_data;
    timing_data.init_n = init_n;
    timing_data.iterations = poisson_steps;
    timing_data.mobility_steps = poisson_timestep;
    timing_data.block_size = block_size;
    timing_data.sleep_time = sleep_time_ns;

    double mobility_timestep = 1e-12;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CSData* cross_sections = (CSData*)malloc(sizeof(CSData)*N_STEPS);
    processCSData(cross_sections, "./src/cross_section.txt");

    
    CSData* d_cross_sections;
    cudaMalloc(&d_cross_sections, N_STEPS * sizeof(CSData));
    cudaMemcpy(d_cross_sections, cross_sections, N_STEPS * sizeof(CSData), cudaMemcpyHostToDevice);
    checkCudaError("CS Alloc");

    const int shared_mem_size_dynamic = 32 * (sizeof(Electron) + sizeof(curandState));
    const int shared_mem_size_naive = block_size * (sizeof(Electron) + sizeof(curandState));

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

    curandState* d_rand_states;
    cudaMalloc(&d_rand_states, 2 * capacity * sizeof(curandState));
    checkCudaError("Alloc rand");
    
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

    int* n_done;
    cudaMalloc(&n_done, sizeof(int));

    int* n_created;
    cudaMalloc(&n_created, sizeof(int));
    
    int* i_global;
    cudaMalloc(&i_global, sizeof(int));
    checkCudaError("Minor alloc");


    cudaExtent extent = make_cudaExtent(Grid_Size.x * sizeof(Cell), Grid_Size.y, Grid_Size.z);
    cudaPitchedPtr d_grid;
    cudaMalloc3D(&d_grid, extent);
    checkCudaError("Grid alloc");

    dim3 dim_block(8,8,8);
    dim3 dim_grid(Grid_Size.x/dim_block.x, Grid_Size.y/dim_block.y, Grid_Size.z/dim_block.z);

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

    cudaEventRecord(start);

    int total_added = 0;
    int total_removed = 0;

    int source_index = 0;
    int destination_index = 0;

    checkCudaError("Setup");

    for (int t = 0; t < poisson_steps; t++)
    {
        int num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
        source_index = (t % 2) * capacity;  // Flips between 0 and capacity.
        destination_index = ((t + 1) % 2) * capacity;  // Opposite of above.

        log(verbose, t, h_electrons, &d_electrons[source_index], n_host, n, capacity);
        cudaMemset(n_done, 0, sizeof(int));
        cudaMemset(i_global, 0, sizeof(int));
        cudaMemcpy(n_created, n_host, sizeof(int), cudaMemcpyHostToDevice);

        /*

        // Grid operations
        resetGrid<<<dim_grid, dim_block>>>(d_grid, Grid_Size);
        // cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
        // checkCudaError("Reset grid");
        particlesToGrid<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
        // cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
        // checkCudaError("Particles to grid");
        updateGrid<<<dim_grid, dim_block>>>(d_grid, Electric_Force_Constant, Grid_Size);
        // cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
        // checkCudaError("Update grid");
        gridToParticles<<<num_blocks_all, block_size>>>(d_grid, &d_electrons[source_index], n, Grid_Size);
        // cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Just a sync for testing
        // checkCudaError("Grid to particles");
        */
        int old_n_host = *n_host;
        /*
        // Simulate
        
        switch(mode){
            case 0:{  // Dynamic
                dynamic<<<num_blocks_pers, block_size, shared_mem_size_dynamic>>>(&d_electrons[source_index], mobility_timestep, n, capacity, &d_rand_states[source_index], poisson_timestep, sleep_time_ns, n_created, n_done, i_global, Sim_Size, d_cross_sections);
                break;
            }
            case 1:{  // CPU Sync
                int last_n = 0;  // The amount of particles present in last run. All of these have been fully simulated.
                while(min(*n_host, capacity) != last_n){  // Stop once nothing new has happened.
                    int num_blocks = (min(*n_host, capacity) - last_n + block_size - 1) / block_size;  // We do not need blocks for the old particles.
                    cpuSync<<<num_blocks, block_size>>>(&d_electrons[source_index], mobility_timestep, n, min(*n_host, capacity), last_n, capacity, &d_rand_states[source_index], poisson_timestep, Sim_Size, d_cross_sections);
                    last_n = min(*n_host, capacity);  // Update last_n to the amount just run. NOT to the amount after this run (we don't know that amount yet).
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);  // Now update to the current amount of particles.
                }
                break;
            }
            case 2:{  // Naive
                for (int mob_t = 1; mob_t <= poisson_timestep; mob_t++){
                    int num_blocks = (min(*n_host, capacity) + block_size - 1) / block_size;
                    naive<<<num_blocks, block_size, shared_mem_size_naive>>>(&d_electrons[source_index], mobility_timestep, n, min(*n_host, capacity), capacity, &d_rand_states[source_index], mob_t, Sim_Size, d_cross_sections);
                    log(verbose, t, h_electrons, d_electrons, n_host, n, capacity);
                    cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
                }
                break;
            }
            case 3:{  // Dynamic Old
                dynamicOld<<<num_blocks_pers, block_size>>>(&d_electrons[source_index], mobility_timestep, n, capacity, &d_rand_states[source_index], poisson_timestep, sleep_time_ns, n_done, i_global, Sim_Size, d_cross_sections);
                break;
            }
        }
        */
        cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
        total_added += *n_host - old_n_host;
        checkCudaError("Mobility steps");

        // Check for overflow
        if (*n_host >= capacity) printf("\n\nOVERFLOW FROM ADDING PARTICLES\n\n\n");

        // Remove dead particles
        cudaMemset(n, 0, sizeof(int));
        num_blocks_all = (min(*n_host, capacity) + block_size - 1) / block_size;
        remove_dead_particles_block<<<num_blocks_all, block_size>>>(&d_electrons[source_index], &d_electrons[destination_index], &d_rand_states[source_index], &d_rand_states[destination_index], n, min(*n_host, capacity));
        old_n_host = *n_host;
        cudaMemcpy(n_host, n, sizeof(int), cudaMemcpyDeviceToHost);
        total_removed += old_n_host - *n_host;
        checkCudaError("Remove dead");

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