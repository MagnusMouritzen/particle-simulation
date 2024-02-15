struct Electron {
    public:
        float3 position;
        float weight;
        float3 velocity;
};

float randomFloat();

int randomInt(int a, int b);

float randomFloat(int a, int b);

void image(int N, Electron* electrons, int iteration);