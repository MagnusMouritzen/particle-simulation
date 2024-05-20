# GIT REPOSITORY
The github repository can be found at: 

<https://github.com/MagnusMouritzen/particle-simulation/tree/final_branch>
# TO RUN THE PROGRAM
The program has been developed to run on the HPC provided by DTU. The following modules are needed to run the code:

```
module load cuda/12.2.2 nvhpc/23.7-nompi gcc/12.3.0-binutils-2.40
```
## PIC
### Compile
```
cd PIC
make
```
### Run
The program `run` takes 8 arguments:
1. Mode
2. Verbose
3. init n
4. max t
5. block size
6. max n
7. sleep time
8. poisson_timestep

```
run [Mode] [Verbose] [init n] [max t] [block size] [max n] [sleep time] [poisson_timestep]
```


## MVP+SchedulerTests
### Compile
```
cd MVP+SchedulerTests
make
```

### Run
The program `run` takes 10 arguments of which the last two are currently not used (set to 0):
1. Mode
2. Verbose
3. init n
4. max t
5. block size
6. max n
7. sleep time
8. split chance

- Run various scheduler tests by picking a mode between 3-15. 

- Run the MVP by picking modes 20-23
```
run [Mode] [Verbose] [init n] [max t] [block size] [max n] [sleep time] [split chance] 0 0
```