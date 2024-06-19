#ifndef FIND_MAX_HPP
#define FIND_MAX_HPP

#include "datastructures.hpp"

int find_max_index_CA(CArray<int> values, int *index);
int find_max_index_GV(GrB_Vector values, GrB_Index *index);

#endif