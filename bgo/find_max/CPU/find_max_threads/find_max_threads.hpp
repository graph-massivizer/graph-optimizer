#ifndef FIND_MAX_THREADS_HPP
#define FIND_MAX_THREADS_HPP

#include "omp.h"
#include <thread>
#include <iostream>
#include <ctime>
#include <vector>
#include "utils.hpp"

#define NUM_THREADS 16

int find_max_threads(CArray<float> v, int *index);

#endif