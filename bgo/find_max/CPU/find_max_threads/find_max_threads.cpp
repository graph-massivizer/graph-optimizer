#include "find_max_threads.hpp"

/*******************************************************************
 * Version 3: Custom thread pool
 ******************************************************************/
void thread_function(unsigned start_i, unsigned end_i, float *v, float &max_value, int &max_index) {
    for (int i = start_i; i < end_i; i++) {
        if (v[i] > max_value) {
            max_value = v[i];
            max_index = i;
        }
    }
}


int find_max_threads(CArray<float> v, int *index) {
    std::vector<float> max_values(NUM_THREADS, -1);
    std::vector<int> max_indices(NUM_THREADS, -1);

    std::vector<std::thread> threads;

    for (unsigned i = 0; i < NUM_THREADS; i++) {
        unsigned start_i = i * v.size / NUM_THREADS;
        unsigned end_i = (i + 1) * v.size / NUM_THREADS;

        threads.emplace_back(thread_function, start_i, end_i, v.data, std::ref(max_values[i]), std::ref(max_indices[i]));
    }

    for (unsigned i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }

    float max_value = -1.;
    int max_index = -1;
    for (int i = 0; i < NUM_THREADS; i++) {
        if (max_values[i] > max_value) {
            max_value = max_values[i];
            max_index = max_indices[i];
        }
    }

    *index = max_index;
    return 0;
}