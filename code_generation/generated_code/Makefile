# -------------------------------------------------
# CUDA location detection
# -------------------------------------------------

CUDA_HOME ?= $(shell dirname $(shell dirname $(shell which nvcc 2>/dev/null)))

ifeq ($(CUDA_HOME),)
  ifeq ($(wildcard /usr/local/cuda),/usr/local/cuda)
    CUDA_HOME := /usr/local/cuda
  else ifeq ($(wildcard /opt/cuda),/opt/cuda)
    CUDA_HOME := /opt/cuda
  else
    $(error Could not find CUDA. Please set CUDA_HOME manually.)
  endif
endif

CUDA_INCLUDES := -I$(CUDA_HOME)/include
CUDA_LIBDIR   := -L$(CUDA_HOME)/lib64
CUDA_RPATH    := -Wl,-rpath,$(CUDA_HOME)/lib64
CUDA_LIBS     := -lcudart

$(info Using CUDA at $(CUDA_HOME))

# -------------------------------------------------
# Compilers
# -------------------------------------------------

CPP_COMPILER = g++

# -------------------------------------------------
# Flags
# -------------------------------------------------

CPP_STD   = -std=c++20
CPP_FLAGS = -O3 -march=native -Wall -Wextra -fPIC -fopenmp

# -------------------------------------------------
# Includes & libraries
# -------------------------------------------------

CPP_INCLUDES = -I/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/include -I/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/include/gap $(CUDA_INCLUDES)
CPP_LIBDIR   = -L/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/lib
CPP_LIBS     = -lCPP_Joules -lgraphblas -llagraph -lm
CPP_RPATH    = -Wl,-rpath,/home/duncan/Nextcloud/Documents/PhD/graph-optimizer/lib

BGOS = /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/pr/GPU/vertex_pull /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_max/GPU/find_max_gpu /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/bfs/GPU/vertex_push /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_path/CPU/fp_ca

PR_LIB = /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/pr/GPU/vertex_pull/libvertex_pull.a 
FIND_MAX_LIB = /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_max/GPU/find_max_gpu/libfind_max_gpu.a 
BFS_LIB = /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/bfs/GPU/vertex_push/libvertex_push.a 
FIND_PATH_LIB = /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_path/CPU/fp_ca/libfp_ca.a 

LIBS = $(PR_LIB) $(FIND_MAX_LIB) $(BFS_LIB) $(FIND_PATH_LIB)

# -------------------------------------------------
# Targets
# -------------------------------------------------

all: main

main: main.o $(LIBS)
	$(CPP_COMPILER) $(CPP_STD) $(CPP_FLAGS) $(CPP_INCLUDES) \
	$^ $(CPP_LIBDIR) $(CPP_LIBS) $(CPP_RPATH) \
	$(CUDA_LIBDIR) $(CUDA_LIBS) $(CUDA_RPATH) \
	-o $@

main.o: main.cpp
	$(CPP_COMPILER) $(CPP_STD) $(CPP_FLAGS) $(CPP_INCLUDES) -c $< -o $@


$(PR_LIB):
	$(MAKE) -C /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/pr/GPU/vertex_pull lib

$(FIND_MAX_LIB):
	$(MAKE) -C /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_max/GPU/find_max_gpu lib

$(BFS_LIB):
	$(MAKE) -C /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/bfs/GPU/vertex_push lib

$(FIND_PATH_LIB):
	$(MAKE) -C /home/duncan/Nextcloud/Documents/PhD/graph-optimizer/bgo/find_path/CPU/fp_ca lib


clean:
	rm -f main main.o
	for d in $(BGOS); do $(MAKE) -C $$d clean; done

.PHONY: all clean