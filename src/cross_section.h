#pragma once

#include <iostream>
#include <fstream>
#include <string>

using namespace std;


struct CSData
{
    public:
        double energy;
        int split_chance;
        int remove_chance;
};

void ProcessCSData(CSData* cross_sections, int nsteps, string path_to_csdata);

