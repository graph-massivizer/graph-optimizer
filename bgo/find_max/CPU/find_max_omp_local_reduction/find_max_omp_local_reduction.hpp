#ifndef FIND_MAX_OMP_LOCAL_REDUCTION_HPP
#define FIND_MAX_OMP_LOCAL_REDUCTION_HPP

#include "omp.h"
#include <thread>
#include <iostream>
#include <ctime>
#include <vector>
#include "utils.hpp"

#define NUM_THREADS 16

int find_max_omp_local_reduction(CArray<float> v, int *index);

#endif