#pragma once

#include "electron.h"

#define cell_size 1
#define epsilon0 8.8541878176E-12
#define pi 3.1415926536

extern int3 Grid_Size;
extern float3 Sim_Size;
extern double Electric_Force_Constant;


struct Cell {
    public:
        double charge;
        float3 acceleration;
};
