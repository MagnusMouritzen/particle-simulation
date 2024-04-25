#include "cross_section.h"

void ProcessCSData(CSData* cross_sections, int nsteps, string path_to_csdata) {
    ifstream ReadIn(path_to_csdata);

    // Check if file is opened successfully
    if (!ReadIn) {
        cerr << "Failed to open file: " << path_to_csdata << endl;
        return;
    }

    for (int i = 0; i < nsteps; i++) {
        if (ReadIn >> cross_sections[i].energy >> cross_sections[i].split_chance >> cross_sections[i].remove_chance) {
            continue;
        } else {
            cerr << "Failed to read data for step " << i << endl;
            break;
        }
    }
    // for (int i = 0; i < nsteps; i++) {
    //     ReadIn >> cross_sections[i].energy;
    //     ReadIn >> cross_sections[i].split_chance;
    //     ReadIn >> cross_sections[i].remove_chance;
    //     cout<< cross_sections[i].energy<<endl;
    //     cout<< cross_sections[i].split_chance<<endl;
    //     cout<< cross_sections[i].remove_chance<<endl;
    //     // printf("CS: %d \n", cross_sections[0].split_chance);
    // }
}
