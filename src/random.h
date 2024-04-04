#include <curand.h>
#include <curand_kernel.h>

__device__ void newRandState(curandState* d_rand_states, int i, int seed);

__device__ float randFloat(curandState* state, float min, float max);

__device__ int randInt(curandState* state, int min, int max);

__global__ void setup_rand(curandState* d_rand_states);