#pragma once

#include <tuple>
#include <stdio.h>

using namespace std;

#define DEAD -2
#define electron_charge -1.602176487e-19

struct Electron {
    public:
        float3 position;
        float weight;
        int creator;
        float3 velocity;
        float3 acceleration;
        int timestamp;

        __host__ void print(){
            printf("(%.6f, %.6f, %.6f) (%.6f, %.6f, %.6f) (%.6f) [%d]\n", position.x, position.y, position.z, velocity.x, velocity.y, velocity.z, weight, timestamp);
        }

        __host__ void print(int i){
            printf("%d: ", i);
            print();
        }

        __host__ tuple<int, float, float, float, float, float, float, float> getKey() const {
            return make_tuple(timestamp, weight, position.y, position.x, position.z, velocity.y, velocity.x, velocity.z);
        }

        __host__ bool operator<(const Electron& other){
            return getKey() < other.getKey();
        }

        __host__ bool operator==(const Electron& other){
            return getKey() == other.getKey();
        }

        __host__ bool operator!=(const Electron& other) {
            return !(*this == other);
        }
};