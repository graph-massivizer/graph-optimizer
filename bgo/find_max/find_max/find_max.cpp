#include <cstddef>

#include "GraphBLAS.h"

#include "datastructures.hpp"

#include "find_max.hpp"

int find_max_index_CA(CArray<int> values, int *index) {
    int max_value = 0;
    for (size_t i = 0; i < values.size; i++) {
        if (values.data[i] <= max_value) { continue; }

        max_value = values.data[i];
        *index = i;
    }
    return 0;
}

int find_max_index_GV(GrB_Vector values, GrB_Index *index) {
    GrB_Index nvals;
    GrB_Vector_nvals(&nvals, values);
    int value;
    int max_value = 0;
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Info info = GrB_Vector_extractElement_INT32(&value, values, i);
        if (info != GrB_SUCCESS) { continue; }
        if (value <= max_value) { continue; }

        max_value = value;
        *index = i;
    }
    return 0;
}
