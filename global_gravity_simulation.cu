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

__global__ static void update(Electron* electrons, float deltaTime) {
    int i = threadIdx.x;
    electrons[i].velocity.y -= 9.82 * deltaTime;
    electrons[i].position.y += electrons[i].velocity.y * deltaTime;
    if (electrons[i].position.y <= 0){
        electrons[i].position.y = -electrons[i].position.y;
        electrons[i].velocity.y = -electrons[i].velocity.y;
    }
}

void globalGravityRun(int N) {

    Electron* electrons_host = (Electron *)malloc(N * sizeof(Electron));
    for(int i=0; i<N; i++) {
        electrons_host[i].position = make_float3(randomFloat(0,5), randomFloat(10,50), 1.0);
        electrons_host[i].weight = 1.0;
    }

    Electron* electrons;
    cudaMalloc(&electrons, N * sizeof(Electron));

    cudaMemcpy(electrons, electrons_host, N * sizeof(Electron), cudaMemcpyHostToDevice);

    printf("Time %d, position %.6f, velocity %.6f\n", 0, electrons_host[0].position.y, electrons_host[0].velocity.y);
    for (int i = 1; i < 101; i++){
        update<<<1, N>>>(electrons, 0.1);

        if (i % 5 == 0){
            cudaMemcpy(electrons_host, electrons, N * sizeof(Electron), cudaMemcpyDeviceToHost);
            printf("Time %d, position %.6f, velocity %.6f\n", i, electrons_host[0].position.y, electrons_host[0].velocity.y);
        }
    }

    

    
}
