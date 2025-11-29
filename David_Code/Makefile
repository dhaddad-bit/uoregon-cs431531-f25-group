# --- Compilers ---
# Use standard names so they pick up the 'module load' versions
CC = gcc
CPP = g++
NVCC = nvcc

# --- Flags ---
# C compilation flags
CFLAGS = -O3 -fopenmp -lm -std=c11
# C++ compilation flags
CCFLAGS = -O3 -fopenmp -lm -std=c++11
# CUDA compilation flags (fix arch if needed, sm_70 for V100, sm_80 for A100)
# We use $(CUDA_HOME) which is set by 'module load cuda'
NVCCFLAGS = -O3 -gencode arch=compute_80,code=sm_80 -std=c++11

# --- Includes and Libs ---
# Use the environment variable CUDA_HOME provided by the module
CUDA_PATH ?= $(CUDA_HOME)
NVCCINC = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm -fopenmp

# --- Files ---
EXEC = spmv

# Note: We now look for .cc files for main
C_SRC = common.c mmio.c
CC_SRC = $(wildcard *.cc)
CU_SRC = spmv.cu

# Generate object file names
C_OBJS = $(C_SRC:.c=.o)
CC_OBJS = $(CC_SRC:.cc=.o)
CU_OBJS = $(CU_SRC:.cu=.o)

OBJS = $(C_OBJS) $(CC_OBJS) $(CU_OBJS)

# --- Rules ---

all: $(EXEC)

# Link everything using the C++ compiler
$(EXEC): $(OBJS)
	$(CPP) $(CCFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

# Compile C files (mmio.c, common.c)
%.o : %.c
	$(CC) $(CFLAGS) $(NVCCINC) -c $< -o $@

# Compile C++ files (main.cc)
%.o : %.cc
	$(CPP) $(CCFLAGS) $(NVCCINC) -c $< -o $@

# Compile CUDA files (spmv.cu)
%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINC) -c $< -o $@

clean:
	rm -f *.o $(EXEC)