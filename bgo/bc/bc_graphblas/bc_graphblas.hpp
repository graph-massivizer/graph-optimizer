#ifndef BC_GRAPHBLAS_HPP
#define BC_GRAPHBLAS_HPP

#include "GraphBLAS.h"

#include "datastructures.hpp"

int bc_graphblas(GrB_Matrix G, CArray<GrB_Index> sources, GrB_Vector *centrality);

#endif