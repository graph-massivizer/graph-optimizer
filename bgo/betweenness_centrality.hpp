#ifndef BC_HPP
#define BC_HPP

#include <vector>
#include "GraphBLAS.h"

int BC_brandes(float *bc, std::vector<int> *G, int *sources, int num_sources, int num_verts);

int BC_naive(float *bc, std::vector<int> *G, int *sources, int num_sources, int num_verts);

GrB_Info BC_GB(GrB_Vector *delta, GrB_Matrix A, GrB_Index *s, GrB_Index nsver);

#endif