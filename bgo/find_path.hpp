#ifndef FIND_PATH_HPP
#define FIND_PATH_HPP

#include <vector>
#include "GraphBLAS.h"

std::vector<int> find_path(int *parent, int start, int end);
std::vector<int> find_path(GrB_Vector parent, int start, int end);

#endif