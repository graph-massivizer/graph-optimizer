#include "fp_gb.hpp"

int find_path(GrB_Vector parents, int start, int end, int_vector path) {
    int prev;
    int current = start;
    while (current != end) {
        path.push_back(current);
        prev = current;
        GrB_Vector_extractElement_INT32(&current, parents, current);
        if (current == prev) {
            /* No path exists. */
            path.push_back(-1);
            return 0;
        }
    }

    path.push_back(end);
    return 0;
}