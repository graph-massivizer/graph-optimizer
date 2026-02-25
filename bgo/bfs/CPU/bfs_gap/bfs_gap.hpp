#ifndef BFS_GAP_HPP
#define BFS_GAP_HPP

#include <iostream>
#include <vector>

#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "sliding_queue.h"
#include "timer.h"
#include "datastructures.hpp"

int bfs_gap(CSR &G, int source, CArray<int> *level, CArray<int> *parent);

#endif