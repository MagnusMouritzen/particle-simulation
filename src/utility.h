#pragma once
#include <limits>
#include <chrono>
#include <string>
#include <vector>
#include <tuple>
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::nanoseconds;

struct Electron {
    public:
        float3 position;
        float weight;
        int creator;
        float3 velocity;
        int timestamp;
        bool live;

        __host__ void print(){
            printf("(%.6f, %.6f) (%.6f, %.6f) (%.6f) [%d]\n", position.x, position.y, velocity.x, velocity.y, weight, timestamp);
        }

        __host__ void print(int i){
            printf("%d: (%.6f, %.6f) (%.6f, %.6f) (%.6f) [%d]\n", i, position.x, position.y, velocity.x, velocity.y, weight, timestamp);
        }

        __host__ tuple<int, float, float, float, float, float, float, float> getKey() const {
            return make_tuple(timestamp, weight, position.y, position.x, position.z, velocity.y, velocity.x, velocity.z);
        }

        __host__ bool operator<(const Electron& other){
            /*if (timestamp != other.timestamp) return timestamp < other.timestamp;
            if (weight != other.weight) return weight < other.weight;
            if (position.y != other.position.y) return position.y < other.position.y;
            if (position.x != other.position.x) return position.x < other.position.x;
            if (position.z != other.position.z) return position.z < other.position.z;
            if (velocity.y != other.velocity.y) return velocity.y < other.velocity.y;
            if (velocity.x != other.velocity.x) return velocity.x < other.velocity.x;
            if (velocity.z != other.velocity.z) return velocity.z < other.velocity.z;
            return true;*/
            return getKey() < other.getKey();
        }

        __host__ bool operator==(const Electron& other){
            // return getKey() == other.getKey();
            return timestamp == other.timestamp &&
            abs(weight - other.weight) < 0.00001 &&
            abs(position.x - other.position.x) < 0.00001 &&
            abs(position.y - other.position.y) < 0.00001 &&
            abs(position.z - other.position.z) < 0.00001 &&
            abs(velocity.x - other.velocity.x) < 0.00001 &&
            abs(velocity.y - other.velocity.y) < 0.00001 &&
            abs(velocity.z - other.velocity.z) < 0.00001;
        }

        __host__ bool operator!=(const Electron& other) {
            return !(*this == other);
        }
};

struct TimingData{
    double time;
    int init_n;
    int iterations;
    int block_size;
    int sleep_time;
    int final_n;
    float split_chance;
    string function;
};

struct RunData{
    TimingData timing_data;
    int final_n;
    Electron* electrons;
};

float randomFloat();

int randomInt(int a, int b);

float randomFloat(int a, int b);

void image(int N, Electron* electrons, int iteration);

chrono::time_point<high_resolution_clock> start_cpu_timer();

double end_cpu_timer(chrono::time_point<high_resolution_clock> start);

void printCSV(const vector<TimingData>& data, string filename);
