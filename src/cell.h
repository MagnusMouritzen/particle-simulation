float cell_size = 8.3e-3;
int3 grid_size = make_int3(512, 512, 512);
float3 sim_size = make_float3(grid_size.x * cell_size, grid_size.y * cell_size, grid_size.z * cell_size);
double epsilon0 = 8.8541878176E-12;
double pi = 3.1415926536;
double electric_force_constant = 1 / (4 * pi * epsilon0 * cell_size * cell_size);


struct Cell {
    public:
        double charge;
        double3 acceleration;
};
