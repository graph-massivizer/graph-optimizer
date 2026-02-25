#include <cstddef>

#include "datastructures.hpp"

#include "find_max_ca.hpp"

int find_max_index_CA(CArray<float> values, int *index) {
    float max_value = 0;
    for (size_t i = 0; i < values.size; i++) {
        if (values.data[i] <= max_value) { continue; }

        max_value = values.data[i];
        *index = i;
    }
    return 0;
}
