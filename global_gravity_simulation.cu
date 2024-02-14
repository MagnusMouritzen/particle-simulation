#include <cuda_runtime.h>
#include <stdio.h>
#include "utility.h"
#include "global_gravity_simulation.h"


struct Electron {
    public:
        float3 position;
        float weight;
        float3 velocity;
};

__global__ static void update(Electron* electrons) {
    int i = threadIdx.x;

}

void globalGravityRun(int N) {

    Electron* electrons_host = (Electron *)malloc(N * sizeof(Electron));
    for(int i=0; i<N; i++) {
        electrons_host[i].position = make_float3(randomFloat(0,5), randomFloat(0,5), 1.0);
        electrons_host[i].velocity = make_float3(0.0, 1.0, 0.0);
        electrons_host[i].weight = 1.0;
    }
    printf("%.6f \n", electrons_host[0].position.y);

    Electron* electrons;
    cudaMalloc(&electrons, N * sizeof(Electron));

    cudaMemcpy(electrons, electrons_host, N * sizeof(Electron), cudaMemcpyHostToDevice);
    
    update<<<1, N>>>(electrons);

    cudaMemcpy(electrons_host, electrons, N * sizeof(Electron), cudaMemcpyDeviceToHost);
    
    printf("%.6f \n", electrons_host[0].position.y);
    
}
