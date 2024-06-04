#include <time.h>
#include <stdio.h>
#include <queue>
#include <cstdlib>

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
        REPEAT_1000(val = a[i]; b = val > 0.5;)
    }

    clock_gettime(CLOCK_MONOTONIC, &after);
    double time = (double)(after.tv_sec - before.tv_sec) +
              (double)(after.tv_nsec - before.tv_nsec) * 1e-9;
    printf("float gt: %.6e\n", time);

    return b;
}

int main() {
    queue_ops();
    int_add();
    float_gt();
    return 0;
}
