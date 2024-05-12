#include "random.h" 

__device__ void newRandState(curandState* rand_state, int seed){
    curand_init(39587 + seed, 0, 0, rand_state);  // Keep base seed below 47483647
}

__device__ float randFloat(curandState* state, float min, float max){
    float rand = curand_uniform(state);
    rand *= (max - min);
    rand += min;
    return rand;
}

__device__ int randInt(curandState* state, int min, int max){
    float rand = curand_uniform(state);
    rand *= (max - min + 0.999999);
    rand += min;
    return (int)truncf(rand);
}
