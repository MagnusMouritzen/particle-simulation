This project is a study in efficient implementation of a particle simulation to run on a GPU. It was the bachelor project of Johannes Poulsen and Magnus Mouritzen.

The motivation for the project was to pave the ground for a GPU-based implementation of a PIC MCC particle simulation developed by a research group at DTU space. It explores the various scheduling paradigms given the challenges of a dynamic simulation and proposes optimised kernels for this task.

The version of the project that was submitted can be found at https://github.com/MagnusMouritzen/particle-simulation/tree/final_branch.

# Information
See the report and/or slideshow from the presentation for additional information.

# TO RUN THE PROGRAM
The program has been developed to run on the HPC provided by DTU. The following modules are needed to run the code:

```
module load cuda/12.2.2 nvhpc/23.7-nompi gcc/12.3.0-binutils-2.40
```
### Compile
```
make
```
### Run
The program `run` takes 8 arguments:
1. Mode (The scheduler to use. 30=Dynamic, 31=CPU Sync, 32=Naive, 33=Dynamic Old).
2. Verbose (0 for no verbosity. Another number determines how often the state of the system is printed. 1 for every mobility step, 2 for every second, etc.).
3. init n (The initial amount of particles).
4. max t (Poisson steps).
5. block size.
6. max n (Capacity for particles).
7. sleep time (Used for some barriers, but largely irrelevant).
8. poisson_timestep (Mobility steps for each poisson step).

```
run [Mode] [Verbose] [init n] [max t] [block size] [max n] [sleep time] [poisson_timestep]
```
