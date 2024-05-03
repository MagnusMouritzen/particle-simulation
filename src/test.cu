#include <algorithm>
#include "multiply_simulation.h"
#include "test.h"
using namespace std;

void runBenchmark(){
    vector<TimingData> data;
    int init_ns[] = {100000};
    int block_sizes[] = {128,256,512,1024};
    int max_ts[] = {10000};
    int max_ns[] = {10000000};
    int functions[] = {0,1,2,3,4,5,6,8,9,10,11,12};
    int sleep_times[] = {100};

    for(int init_n : init_ns){
        for(int block_size : block_sizes){
            for(int max_t : max_ts){
                for(int max_n : max_ns){
                    for(int sleep_time : sleep_times){
                        for(int function : functions){
                            RunData run_data = multiplyRun(init_n, max_n, max_t, function, 0, block_size, sleep_time, 0.01);
                            data.push_back(run_data.timing_data);
                            free(run_data.electrons);
                        }
                    }
                }
            }
        }
    }
    printCSV(data, "out/data/schedulers.csv");
}

void runUnitTest(int init_n, int max_n, int max_t, int verbose, int block_size, int sleep_time){
    // How I ran it:
    // run test 0 1 200 256 10000000 100

    int base_function = 0;
    int test_functions[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    //int test_functions[] = {4};
    int amnt = sizeof(test_functions) / sizeof(int);
    bool broken[amnt];
    int final_ns[amnt];

    RunData base_run_data = multiplyRun(init_n, max_n, max_t, base_function, 0, block_size, sleep_time, 1);
    Electron* base_electrons = base_run_data.electrons;
    int base_final_n = base_run_data.final_n;
    printf("Sorting base...\n");
    sort(base_electrons, base_electrons + base_final_n);
    printf("Done sorting base\n");

    for(int fi = 0; fi < amnt; fi++){
        int function = test_functions[fi];
        broken[fi] = false;
        RunData run_data = multiplyRun(init_n, max_n, max_t, function, 0, block_size, sleep_time, 1);
        Electron* electrons = run_data.electrons;
        int final_n = run_data.final_n;
        final_ns[fi] = final_n;
        if (final_n != base_final_n){
            printf("\n\nFinal n does not match in %d. Base: %d, test: %d\n\n\n", function, base_final_n, final_n);
            broken[fi] = true;
            continue;
        }
        
        printf("Sorting %d...\n", function);
        sort(electrons, electrons + final_n);
        printf("Done sorting\n");
        for(int i = 0; i < final_n; i++){
            if (base_electrons[i] != electrons[i]){
                printf("Mismatch in %d!\n", function);
                base_electrons[i].print(i);
                electrons[i].print(i);
                broken[fi] = true;
                break;
            }
            else if (verbose != 0 && i % verbose == 0) base_electrons[i].print(i);
        }
    }

    printf("\nTests done with following results as compared to %d (%d):\n", base_function, base_final_n);
    for(int fi = 0; fi < amnt; fi++){
        printf("%d: ", test_functions[fi]);
        if (broken[fi]){
            printf("failure");
        }
        else{
            printf("success");
        }
        printf(" (%d)\n", final_ns[fi]);
    }
}