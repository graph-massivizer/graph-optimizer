PYTHON			=	python3

CPP_COMPILER	=	g++ -std=c++11
CPP_FLAGS       =	-O3 -march=native -Wall -Wextra -fPIC
CPP_INCLUDES    =	-I$(HOME)/local_install/usr/local/include/suitesparse -I../../../include
CPP_LIBS        =	-llagraph -lgraphblas -lm -Wl,-rpath,$(HOME)/local_install/usr/local/lib64 -L$(HOME)/local_install/usr/local/lib64

HEADER  = $(notdir $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))).hpp
SOURCES = $(filter-out bench.cpp, $(wildcard *.c *.cpp))
OBJECTS = $(patsubst %.c,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))


all: bench

bench.cpp: $(HEADER)
	$(PYTHON) ../../../autobench/gen_bench_code.py $@ $^ '$(CPP_INCLUDES)'

%.o: %.cpp
	$(CPP_COMPILER) $(CPP_FLAGS) -c $(CPP_INCLUDES) $< -o $@

bench: bench.cpp $(OBJECTS)
	$(CPP_COMPILER) $(CPP_FLAGS) $(CPP_INCLUDES) $(CPP_LIBS) $^ -o $@

clean:
	rm -f bench bench.cpp *.o

.PHONY: all clean
