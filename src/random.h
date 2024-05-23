#include <curand.h>
#include <curand_kernel.h>

__device__ void newRandState(curandState* rand_state, int sequence);

__device__ float randFloat(curandState* state, float min, float max);

__device__ int randInt(curandState* state, int min, int max);