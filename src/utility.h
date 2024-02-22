#include <limits>
using namespace std;

struct Electron {
    public:
        float3 position;
        float weight;
        float3 velocity;
        int timestamp = numeric_limits<int>::max();
};

float randomFloat();

int randomInt(int a, int b);

float randomFloat(int a, int b);

void image(int N, Electron* electrons, int iteration);