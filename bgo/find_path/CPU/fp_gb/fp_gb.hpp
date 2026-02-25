#ifndef FIND_PATH_GB_HPP
#define FIND_PATH_GB_HPP

#include <vector>
#include "GraphBLAS.h"
#include "datastructures.hpp"

int find_path(GrB_Vector parents, int start, int end, int_vector path);

#endif