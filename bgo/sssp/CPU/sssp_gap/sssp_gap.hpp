#ifndef SSSP_GAP_HPP
#define SSSP_GAP_HPP

#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"
#include "datastructures.hpp"
#include "benchmark.h"

int SSSP_gap(CSR &G, int source, int delta, CArray<int> *distances);

#endif