#include "multiply_simulation.h"
#include "test.h"
using namespace std;

void runTests(){
    vector<TimingData> data;
    int init_ns[] = {10000};
    int block_sizes[] = {128,256,512};
    int max_ts[] = {10000};
    int max_ns[] = {1000000};
    int functions[] = {1,2,3,4,5,6,7,8,9,10,11,12};
    int sleep_times[] = {100};

    for(int init_n : init_ns){
        for(int block_size : block_sizes){
            for(int max_t : max_ts){
                for(int max_n : max_ns){
                    for(int sleep_time : sleep_times){
                        for(int function : functions){
                            data.push_back(multiplyRun(init_n, max_n, max_t, function, 0, block_size, sleep_time));
                        }
                    }
                }
            }
        }
    }
    printCSV(data, "out/data.csv");
}