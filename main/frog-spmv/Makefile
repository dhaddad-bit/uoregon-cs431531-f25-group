# --- Compilers ---
# Use standard names so they pick up the 'module load' versions
CC = gcc
CPP = g++
NVCC = nvcc

# Specify subdirectory to search for headers and source files

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin
INC_DIR := include

$(shell mkdir -p $(OBJ_DIR) $(BIN_DIR))

VPATH := $(SRC_DIR):$(SRC_DIR)/util:$(INC_DIR):$(INC_DIR)/helpers:$(INC_DIR)/NaiveCSR5


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
NVCCINC = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -Ihelpers -INaiveCSR5

INC_FLAGS = -I$(INC_DIR) \
            -I$(INC_DIR)/helpers \
            -I$(INC_DIR)/NaiveCSR5 \
            -I$(CUDA_PATH)/include \
            -I$(CUDA_PATH)/samples/common/inc

LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lm -fopenmp

# --- Files ---
EXEC = $(BIN_DIR)/spmv

# Note: We now look for .cc files for main
C_SRC = util/common.c util/mmio.c
CC_SRC = main.cc csr5_conversion.cc
CU_SRC = spmv.cu csr5_spmv.cu

# Generate object file names
C_OBJS = $(addprefix $(OBJ_DIR)/,$(notdir $(C_SRC:.c=.o)))
CC_OBJS = $(addprefix $(OBJ_DIR)/,$(notdir $(CC_SRC:.cc=.o)))
CU_OBJS = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_SRC:.cu=.o)))

OBJS = $(C_OBJS) $(CC_OBJS) $(CU_OBJS)

# --- Rules ---

all: $(EXEC)

# Link everything using the C++ compiler
$(EXEC): $(OBJS)
	$(CPP) $(CCFLAGS) $(OBJS) -o $(EXEC) $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CPP) $(CCFLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@



$(OBJ_DIR)/%.o: $(SRC_DIR)/util/%.c
	$(CC) $(CFLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/util/%.cc
	$(CPP) $(CCFLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/util/%.cu
	$(NVCC) $(NVCCFLAGS) $(INC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJ_DIR)/*.o $(EXEC)