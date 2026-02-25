#include "fp_ca.hpp"

int find_path(CArray<int> parents, int start, int end, int_vector &path) {
    int current = start;
    while (current != end) {
        path.push_back(current);
        if (current == parents.data[current]) {
            break;
        }
        current = parents.data[current];
    }
    path.push_back(end);

    return 0;
}
