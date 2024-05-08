#include "cross_section.h"

void processCSData(CSData* cross_sections, string path_to_csdata) {
    ifstream ReadIn(path_to_csdata);

    // Check if file is opened successfully
    if (!ReadIn) {
        cerr << "Failed to open file: " << path_to_csdata << endl;
        return;
    }

    for (int i = 0; i < N_STEPS; i++) {
        if (ReadIn >> cross_sections[i].split_chance >> cross_sections[i].remove_chance) {
            continue;
        } else {
            cerr << "Failed to read data for step " << i << endl;
            break;
        }
    }
    // for (int i = 0; i < N_STEPS; i++) {
    //     ReadIn >> cross_sections[i].split_chance;
    //     ReadIn >> cross_sections[i].remove_chance;
    //     cout<< cross_sections[i].energy<<endl;
    //     cout<< cross_sections[i].split_chance<<endl;
    //     cout<< cross_sections[i].remove_chance<<endl;
    //     // printf("CS: %d \n", cross_sections[0].split_chance);
    // }
}

__device__ int energyToIndex(double energy){
    int energy_index = trunc(log10(energy));
    return (energy_index < 0) ? 0 : ((energy_index > 10) ? 10 : energy_index);
}
