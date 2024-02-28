#include <vector>
#include "find_path.hpp"
#include "GraphBLAS.h"

std::vector<int> find_path(int *parent, int start, int end) {
    std::vector<int> path;

    int current = start;
    while (current != end) {
        path.push_back(current);
        current = parent[current];
    }
    path.push_back(end);

    return path;
}

std::vector<int> find_path(GrB_Vector parent, int start, int end) {
    std::vector<int> path;

    int current = start;
    while (current != end) {
        path.push_back(current);
        GrB_Vector_extractElement_INT32(&current, parent, current);
    }
    path.push_back(end);

    return path;
}