#include "find_max_omp_custom_reduction.hpp"

struct MaxValue {
    float value;
    int index;
};

/*******************************************************************
 * Version 1: OpenMP with custom reduction
 ******************************************************************/
MaxValue& max(MaxValue& a, MaxValue& b) {
    return a.value > b.value ? a : b;
}

#pragma omp declare reduction(customMax:MaxValue:omp_out=max(omp_out, omp_in))

int find_max_omp_custom_reduction(CArray<float> v, int *index) {
    MaxValue max = {-1., -1};
    #pragma omp parallel for reduction(customMax:max)
    for (unsigned i = 0; i < v.size; i++) {
        if (v.data[i] > max.value) {
            max.value = v.data[i];
            max.index = i;
        }
    }

    *index = max.index;

    return 0;
}