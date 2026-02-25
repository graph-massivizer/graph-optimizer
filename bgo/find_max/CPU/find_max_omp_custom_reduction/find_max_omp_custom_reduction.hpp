#ifndef FIND_MAX_OMP_CUSTOM_REDUCTION_HPP
#define FIND_MAX_OMP_CUSTOM_REDUCTION_HPP

#include "omp.h"
#include <thread>
#include <iostream>
#include <ctime>
#include <vector>
#include "utils.hpp"

#define NUM_THREADS 16

int find_max_omp_custom_reduction(CArray<float> v, int *index);

#endif