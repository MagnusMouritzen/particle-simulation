# Simple setup of a project with CUDA, Thrust and C++

# CUDA compiler
CCC = nvcc
# C++ compiler
CXX = nvc++

# Flags
CFLAGS = -O3 -std=c++11
# CUDA flags
CUDAFLAGS = -arch=sm_70
# sm_70 is the architecture of the GPU, it can be changed to match the GPU you are using.
# Currently we're using a Tesla V100, which has the architecture sm_70.

all: main

main: simp_simulation.cu
	$(CCC) $(CFLAGS) $(CUDAFLAGS) -o simp_simulation simp_simulation.cu

# Clean
clean:
	rm -f simp_simulation

remake: clean simp_simulation

