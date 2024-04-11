#include "cell.h"

int3 Grid_Size = make_int3(512, 512, 512);
float3 Sim_Size = make_float3(Grid_Size.x * cell_size, Grid_Size.y * cell_size, Grid_Size.z * cell_size);
double Electric_Force_Constant = (electron_charge*electron_charge) / (4 * pi * epsilon0 * cell_size * cell_size * electron_mass);