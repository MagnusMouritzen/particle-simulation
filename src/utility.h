#pragma once
#include <limits>
#include <chrono>
#include <string>
#include <vector>
#include <tuple>
#include <cuda_runtime.h>
#include <stdexcept>
#include "electron.h"
using namespace std;
using chrono::high_resolution_clock;
using chrono::duration_cast;
using chrono::nanoseconds;

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

void image(int N, Electron* electrons, int iteration);

chrono::time_point<high_resolution_clock> start_cpu_timer();

double end_cpu_timer(chrono::time_point<high_resolution_clock> start);

void printCSV(const vector<TimingData>& data, string filename);

void checkCudaError();

void log(int verbose, int t, Electron* electrons_host, Electron* electrons, int* n_host, int* n, int capacity);