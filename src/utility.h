#include <limits>
#include <chrono>
#include <string>
#include <vector>
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::nanoseconds;

struct Electron {
    public:
        float3 position;
        float weight;
        float3 velocity;
        volatile int timestamp = numeric_limits<int>::max();
};

struct TimingData{
    double time;
    int init_n;
    int iterations;
    int block_size;
    string function;
};

float randomFloat();

int randomInt(int a, int b);

float randomFloat(int a, int b);

void image(int N, Electron* electrons, int iteration);

chrono::time_point<high_resolution_clock> start_cpu_timer();

double end_cpu_timer(chrono::time_point<high_resolution_clock> start);

void printCSV(const vector<TimingData>& data, string filename);
