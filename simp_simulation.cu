#include <cuda_runtime.h>
#include <stdio.h>

class Electron {
    public:
        float3 position;
        float weight;
        float3 velocity;
};

float randomFloat() {
    return (float)(rand()) / (float)(RAND_MAX);
}
 
int randomInt(int a, int b)
{
    if (a > b)
        return randomInt(b, a);
    if (a == b)
        return a;
    return a + (rand() % (b - a));
}

float randomFloat(int a, int b)
{
    if (a > b)
        return randomFloat(b, a);
    if (a == b)
        return a;
 
    return (float)randomInt(a, b) + randomFloat();
}

__global__ void init(const Electron* electrons, int n) {

}

__global__ void update(Electron* electrons) {
    int i = threadIdx.x;
    electrons[i].position.y += electrons[i].velocity.y;
}

int main(int argc, char **argv) {

    int N = 1000;
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
