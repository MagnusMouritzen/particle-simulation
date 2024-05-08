#pragma once

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define N_STEPS 11

struct CSData
{
    public:
        float split_chance;
        float remove_chance;
};

void processCSData(CSData* cross_sections, string path_to_csdata);

__device__ int energyToIndex(double energy);
