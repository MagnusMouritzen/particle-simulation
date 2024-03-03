#include "multiply_simulation.h"
#include "test.h"
using namespace std;

void runTests(){
    vector<TimingData> data;
    int init_ns[] = {1, 10000};
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    int max_ts[] = {10000};
    int max_ns[] = {100000};
    int functions[] = {0, 1, 2, 3, 4};

    for(int init_n : init_ns){
        for(int block_size : block_sizes){
            for(int max_t : max_ts){
                for(int max_n : max_ns){
                    for(int function : functions){
                        data.push_back(multiplyRun(init_n, max_n, max_t, function, 0, block_size));
                    }
                }
            }
        }
    }
    printCSV(data, "out/data.csv");
}