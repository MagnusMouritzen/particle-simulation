
inline int cell_size = 8.3e-3;
inline int3 grid_size = make_int3(512, 512, 512);
inline float3 sim_size = make_float3(grid_size.x * 8.3e-3, grid_size.y * 8.3e-3, grid_size.z * 8.3e-3);

struct Cell {
    public:
        double charge;
        float3 forces;
};
