#include <time.h>
#include <stdio.h>
#include <queue>
#include <cstdlib>
#include <vector>
#include "../include/BinaryHeap.hpp"

#define REPEAT_10(x) x x x x x x x x x x
#define REPEAT_100(x) REPEAT_10(REPEAT_10(x))
#define REPEAT_1000(x) REPEAT_10(REPEAT_100(x))
#define REPEAT_10000(x) REPEAT_10(REPEAT_1000(x))

int queue_ops() {
    register int a = 42;
    std::queue<int> queue;

    struct timespec before_push, after_push, after_front, after_pop;
    double time_push, time_front, time_pop;

    for (register int i = 0; i < 100000; i++) {
        clock_gettime(CLOCK_MONOTONIC, &before_push);
        REPEAT_10000(queue.push(a););
        clock_gettime(CLOCK_MONOTONIC, &after_push);
        REPEAT_10000(a = queue.front(););
        clock_gettime(CLOCK_MONOTONIC, &after_front);
        REPEAT_10000(queue.pop(););
        clock_gettime(CLOCK_MONOTONIC, &after_pop);

        time_push += (double)(after_push.tv_sec - before_push.tv_sec) +
              (double)(after_push.tv_nsec - before_push.tv_nsec) * 1e-9;
        time_front += (double)(after_front.tv_sec - after_push.tv_sec) +
              (double)(after_front.tv_nsec - after_push.tv_nsec) * 1e-9;
        time_pop += (double)(after_pop.tv_sec - after_front.tv_sec) +
              (double)(after_pop.tv_nsec - after_front.tv_nsec) * 1e-9;
    }

    printf("push: %.6e\n", time_push);
    printf("front: %.6e\n", time_front);
    printf("pop: %.6e\n", time_pop);

    return a;
}

int int_add() {
    register int a = 0;

    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);

    for (register int i = 0; i < 100000; i++) {
        REPEAT_10000(a++;)
    }

    clock_gettime(CLOCK_MONOTONIC, &after);
    double time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    printf("int add: %.6e\n", time);
    return a;
}

bool float_gt() {
    // Initialize a random array of floats
    float a[1000000];
    register bool b;
    register float val;
    for (int i = 0; i < 1000000; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }

    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);

    for (register int i = 0; i < 1000000; i++) {
        REPEAT_1000(b = 0.1 > 0.5;)
    }

    clock_gettime(CLOCK_MONOTONIC, &after);
    double time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    printf("float gt: %.6e\n", time);

    return b;
}

bool int_neq() {
    register int a = 0;
    register bool b;

    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);

    for (register int i = 0; i < 100000; i++) {
        REPEAT_10000(b = a != 0;)
    }

    clock_gettime(CLOCK_MONOTONIC, &after);
    double time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    printf("int neq: %.6e\n", time);
    return b;
}

int binary_heap() {
    register int N = 1000;

    struct timespec before, after;
    clock_gettime(CLOCK_MONOTONIC, &before);
    for (register int i = 0; i < 1000; i++) {
        BinaryHeap heap1(N);
    }
    clock_gettime(CLOCK_MONOTONIC, &after);

    double insert_max_time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-6;
    printf("binary heap insert max: %.6e\n", insert_max_time);

    double extract_min_time = 0;
    int min = 0;
    for (register int i = 0; i < N; i++) {
        BinaryHeap heap2(N);
        clock_gettime(CLOCK_MONOTONIC, &before);
        REPEAT_1000(min = heap2.extract_min();)
        clock_gettime(CLOCK_MONOTONIC, &after);

        extract_min_time += (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-6;
    }
    printf("binary heap extract min: %.6e\n", extract_min_time);

    // worst case execution time
    double heap_decrease_key_time = 0;
    for (register int i = 0; i < N; i++) {
        BinaryHeap heap3(N);

        clock_gettime(CLOCK_MONOTONIC, &before);
        REPEAT_1000(heap3.decrease_key(i, 0);)
        clock_gettime(CLOCK_MONOTONIC, &after);

        heap_decrease_key_time += (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-6;
    }
    printf("binary heap decrease key: %.6e\n", heap_decrease_key_time);

    return min;
}

std::vector<int> vector_ops() {
    // push_back measurement
    struct timespec before, after;
    double push_back_time = 0;
    for (register int i = 0; i < 100000; i++) {
        std::vector<int> vec;
        clock_gettime(CLOCK_MONOTONIC, &before);
        REPEAT_10000(vec.push_back(i);)
        clock_gettime(CLOCK_MONOTONIC, &after);

        push_back_time += (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    }
    printf("vector push_back: %.6e\n", push_back_time);
}

int main() {
    // queue_ops();
    // int_add();
    // float_gt();
    // binary_heap();
    vector_ops();
    return 0;
}
