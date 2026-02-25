#include "find_max_omp_local_reduction.hpp"

/*******************************************************************
 * Version 2: OpenMP with local reduction
 ******************************************************************/
int find_max_omp_local_reduction(CArray<float> v, int *index) {
    float max_value = -1.;
    int max_index = -1;
    #pragma omp parallel
    {
        float max_value_local = -1.;
        int max_index_local = -1;
        #pragma omp for
        for (unsigned i = 0; i < v.size; i++) {
            if (v.data[i] > max_value_local) {
                max_value_local = v.data[i];
                max_index_local = i;
            }
        }
        #pragma omp critical
        {
            if (max_value_local > max_value) {
                max_value = max_value_local;
                max_index = max_index_local;
            }
        }
    }
    
    *index = max_index;

    return 0;
}