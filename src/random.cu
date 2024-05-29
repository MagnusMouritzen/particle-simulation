#include "random.h" 

__device__ void newRandState(curandState* rand_state, int sequence){
    curand_init(39587, sequence, 0, rand_state);
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

__global__ void setup_rand(curandState* d_rand_states) {
    int i = threadIdx.x+blockDim.x*blockIdx.x;
    newRandState(&d_rand_states[i], i);
}
