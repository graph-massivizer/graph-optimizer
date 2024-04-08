#include "omp.h"
#include <thread>
#include <iostream>
#include <ctime>
#include <vector>

const int NUM_THREADS = 4;

/*******************************************************************
 * Version 1: OpenMP with custom reduction
 ******************************************************************/
struct MaxValue {
    float value;
    int index;
};

MaxValue& max(MaxValue& a, MaxValue& b) {
    return a.value > b.value ? a : b;
}

#pragma omp declare reduction(customMax:MaxValue:omp_out=max(omp_out, omp_in))

int find_max_omp_custom_reduction(float *v, int n) {
    MaxValue max = {-1., -1};
    #pragma omp parallel for reduction(customMax:max)
    for (int i = 0; i < n; i++) {
        if (v[i] > max.value) {
            max.value = v[i];
            max.index = i;
        }
    }

    return max.index;
}

/*******************************************************************
 * Version 2: OpenMP with local reduction
 ******************************************************************/
int find_max_omp_local_reduction(float *v, int n) {
    float max_value = -1.;
    int max_index = -1;
    #pragma omp parallel
    {
        float max_value_local = -1.;
        int max_index_local = -1;
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (v[i] > max_value_local) {
                max_value_local = v[i];
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

    return max_index;
}

/*******************************************************************
 * Version 3: OpenMP with local reduction
 ******************************************************************/
void thread_function(int start_i, int end_i, std::vector<float> const &v, float &max_value, int &max_index) {
    for (int i = start_i; i < end_i; i++) {
        if (v[i] > max_value) {
            max_value = v[i];
            max_index = i;
        }
    }
}

int find_max_threads(std::vector<float> v, int n) {
    std::vector<float> max_indices(NUM_THREADS, -1);
    std::vector<int> max_values(NUM_THREADS, -1);

    std::vector<std::thread> threads;

    int start_i;
    int end_i;

    for (int i = 0; i < NUM_THREADS; i++) {
        start_i = i * n / NUM_THREADS;
        end_i = (i + 1) * n / NUM_THREADS;

        std::thread t(thread_function, start_i, end_i, std::ref(v), std::ref(max_values[i]), std::ref(max_indices[i]));
        threads.push_back(std::move(t));
    }

    for (int i = 0; i < NUM_THREADS; i++) {
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

    return max_index;
}


int main() {
    srand(time(0));
    const int n = 1000000;
    float v[n];

    for (int i = 0; i < n; i++) {
        v[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/n));
    }

    double start_time, time_elapsed;

}