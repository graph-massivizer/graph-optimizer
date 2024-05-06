#ifndef FIND_MAX_GPU_H
#define FIND_MAX_GPU_H

#define BLOCKSIZE 256
#define warpSize 32

int find_max_gpu(float *v, int n);

#endif